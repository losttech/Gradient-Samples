﻿namespace LostTech.Gradient.Samples.GPT2
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using LostTech.Gradient.BuiltIns;
    using Newtonsoft.Json;
    using tensorflow;
    using tensorflow.contrib.training;

    public static class Gpt2Model
    {
        public static HParams DefaultHParams => new HParams(kwargs: new PythonDict<string, object> {
            ["n_vocab"] = 0,
            ["n_ctx"] = 1024,
            ["n_embd"] = 768,
            ["n_head"] = 12,
            ["n_layer"] = 12,
        });

        /// <summary>
        /// Deal with dynamic shape in tensorflow cleanly.
        /// </summary>
        static PythonList<dynamic> ShapeList(ITensor tensor)
        {
            IEnumerable<int?> @static = tensor.shape.as_list();
            dynamic dynamic = tf.shape(tensor);
            var result = new PythonList<dynamic>();
            foreach (object size in @static.Select((size, index) => size ?? (object)dynamic[index]))
                result.Add(size);
            return result;
        }

        static Tensor Softmax(Tensor input, int axis = -1)
        {
            Tensor negative = input - tf.reduce_max(input, axis: new[] { axis }, keepdims: true);
            Tensor exp = tf.exp(negative);
            return exp / tf.reduce_sum(exp, axis: axis, keepdims: true);
        }

        static Tensor GeLU(Tensor input) =>
            ((dynamic)input * 0.5) * (tf.tanh((input + tf.pow(input, 3) * 0.044715) * Math.Sqrt(2 / Math.PI)) + 1);

        /// <summary>
        /// Normalize to mean = 0, std = 1, then do a diagonal affine transform.
        /// </summary>
        static Tensor Norm(Tensor input, object scope, int axis = -1, double epsilon = 1e-5)
        {
            Tensor result = null;
            using (new variable_scope(scope).StartUsing())
            {
                Dimension nState = input.shape[-1];
                Variable g = tf.get_variable("g", new TensorShape(nState), initializer: new constant_initializer(1));
                Variable b = tf.get_variable("b", new TensorShape(nState), initializer: new constant_initializer(0));
                Tensor mean = tf.reduce_mean(input, axis: axis, keepdims: true);
                Tensor s = tf.reduce_mean(tf.square(input - mean), axis: axis, keepdims: true);
                result = (input - mean) * tf.rsqrt(s + epsilon);
                result = result * g + b;
            }
            return result;
        }

        /// <summary>
        /// Reshape the last dimension of input into [n, input.shape[-1]/n]
        /// </summary>
        static Tensor SplitStates(Tensor input, int n)
        {
            var shape = ShapeList(input);
            dynamic reminder = shape.Last() / n;
            shape[shape.Count - 1] = n;
            shape.Add(reminder);
            return tf.reshape_dyn(input, shape);
        }

        /// <summary>
        /// Smash the last two dimensions of input into a single dimension.
        /// </summary>
        static Tensor MergeStates(Tensor input)
        {
            var shape = ShapeList(input);
            shape[shape.Count - 2] = shape[shape.Count - 2] * shape[shape.Count - 1];
            shape.RemoveAt(shape.Count - 1);
            return tf.reshape_dyn(input, shape);
        }

        static Tensor Conv1D(Tensor input, object scope, int nf, double wInitialStDev = 0.02)
        {
            Tensor result = null;
            using (new variable_scope(scope).StartUsing())
            {
                var shape = ShapeList(input);
                var start = shape.Take(shape.Count - 1);
                object nx = shape.Last();
                var wShape = new TensorShape(ValueTuple.Create(1, nx, nf));
                var w = tf.get_variable("w", wShape, initializer: new random_normal_initializer(stddev: wInitialStDev));
                var b = tf.get_variable("b", new TensorShape(nf), initializer: new constant_initializer(0));
                result = tf.reshape_dyn(
                    tf.matmul(
                        tf.reshape_dyn(input, new[] { -1, nx }),
                        tf.reshape(w, new[] { -1, nf })) + b,
                    start.Append(nf).ToArray());
            }
            return result;
        }

        /// <summary>
        /// 1's in the lower triangle, counting from the lower right corner.
        /// Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        /// </summary>
        static Tensor AttentionMask(dynamic nd, dynamic ns, DType dtype = null)
        {
            var i = tf.range(nd)[.., null];
            var j = tf.range(ns);
            var m = i >= j - ns + nd;
            return tf.cast(m, dtype);
        }

        static ValueTuple<Tensor, Tensor> Attention(Tensor input, object scope, int nState, Tensor past = null, dynamic hParams = null)
        {
            if (input.shape.ndims != 3)
                throw new ArgumentException();
            Trace.Assert(nState % (int)hParams.n_head == 0);
            if (!(past is null) && past.shape.ndims != 5)
                throw new ArgumentException();

            Tensor SplitHeads(Tensor x) =>
                // From [batch, sequence, features] to [batch, heads, sequence, features]
                tf.transpose(SplitStates(x, hParams.n_head), new[] { 0, 2, 1, 3 });

            Tensor MergeHeads(Tensor x) =>
                // Reverse of split_heads
                MergeStates(tf.transpose(x, new[] { 0, 2, 1, 3 }));

            Tensor MaskAttentionWeights(Tensor w)
            {
                // w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
                var shape = ShapeList(w);
                object nd = shape[shape.Count - 2];
                object ns = shape[shape.Count - 1];
                var b = AttentionMask(nd, ns, dtype: w.dtype);
                b = tf.reshape_dyn(b, new object[] { 1, 1, nd, ns });
                w = (dynamic)w * b - tf.cast(1e10, w.dtype) * (tf.constant(1.0) - (dynamic)b);
                return w;
            }

            Tensor MultiHeadAttention(Tensor q, Tensor k, Tensor v)
            {
                // q, k, v have shape [batch, heads, sequence, features]
                Tensor w = tf.matmul(q, k, transpose_b: true);
                w *= tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype));

                w = MaskAttentionWeights(w);
                w = Softmax(w);
                return tf.matmul(w, v);
            }

            Tensor attention = null;
            Tensor present = null;
            using (new variable_scope(scope).StartUsing())
            {
                var c = Conv1D(input, "c_attn", nState * 3);
                var qkv = ((IEnumerable)tf.split(c, 3, axis: 2)).Cast<Tensor>().Select(SplitHeads).ToArray();
                var q = qkv[0];
                var k = qkv[1];
                var v = qkv[2];

                present = tf.stack(new[] { k, v }, axis: 1);
                if (!(past is null))
                {
                    var pastKV = tf.unstack(past, axis: 1);
                    k = tf.concat(new[] { pastKV[0], k }, axis: -2);
                    v = tf.concat(new[] { pastKV[1], v }, axis: -2);
                }

                attention = MultiHeadAttention(q, k, v);
                attention = MergeHeads(attention);
                attention = Conv1D(attention, "c_proj", nState);
            }

            return ValueTuple.Create(attention, present);
        }

        static Tensor MLP(Tensor input, string scope, int nState, dynamic hParams = null)
        {
            Tensor result = null;
            using (new variable_scope(scope).StartUsing())
            {
                int nx = input.shape[-1].value;
                var h = GeLU(Conv1D(input, "c_fc", nState));
                result = Conv1D(h, "c_proj", nx);
            }
            return result;
        }

        static ValueTuple<Tensor, Tensor> Block(Tensor input, string scope, Tensor past = null, dynamic hParams = null)
        {
            Tensor result = null;
            Tensor present = null;
            using (new variable_scope(scope).StartUsing())
            {
                int nx = input.shape[-1].value;
                var attentionPresent = Attention(Norm(input, "ln_1"), "attn", nx, past: past, hParams: hParams);
                Tensor attention = attentionPresent.Item1;
                present = attentionPresent.Item2;
                input += (dynamic)attention;
                var m = MLP(Norm(input, "ln_2"), "mlp", nx * 4, hParams: hParams);
                input += m;
                result = input;
            }
            return ValueTuple.Create(result, present);
        }

        public static int?[] PastShape(dynamic hParams = null, int? batchSize = null, int? sequence = null)
        {
            return new int?[]
            {
                batchSize,
                hParams.n_layer,
                2,
                hParams.n_head,
                sequence,
                (int)hParams.n_embd / (int)hParams.n_head,
            };
        }

        /// <summary>
        /// "Add a new axis of given size.
        /// </summary>
        static Tensor ExpandTile(object value, Tensor size)
        {
            Tensor tensor = tf.convert_to_tensor(value, dtype: (DType)null, name: "value");
            int ndims = tensor.shape.rank.Value;
            return tf.tile_dyn(
                tf.expand_dims(tensor, axis: 0),
                multiples: new object[] { size }.Concat(Enumerable.Repeat((object)1, ndims)).ToArray());
        }

        static Tensor PositionsFor(dynamic tokens, Tensor pastLength)
        {
            Tensor batchSize = tf.shape(tokens)[0];
            Tensor nSteps = tf.shape(tokens)[1];
            dynamic stepsRange = tf.range(nSteps, dtype: tf.int32);
            Tensor result = ExpandTile(stepsRange + pastLength, batchSize);
            if (!result.dtype.is_integer)
                throw new InvalidOperationException();
            return result;
        }

        public static Dictionary<string, Tensor> Model(dynamic hParams, Tensor input, dynamic past = null, string scope = "model", object reuse = null)
        {
            var result = new Dictionary<string, Tensor>();
            using (new variable_scope(scope, reuse: reuse).StartUsing())
            {
                var batchSeq = ShapeList(input);
                int batch = batchSeq[0];
                dynamic sequence = batchSeq[1];

                var wpe = tf.get_variable("wpe", new TensorShape((int)hParams.n_ctx, (int)hParams.n_embd), initializer: new random_normal_initializer(stddev: 0.01));
                var wte = tf.get_variable("wte", new TensorShape((int)hParams.n_vocab, (int)hParams.n_embd), initializer: new random_normal_initializer(stddev: 0.02));

                Tensor pastLen = past is null ? tf.constant(0) : tf.shape(past)[^2];
                var h = tf.gather_dyn(wte, input) + tf.gather_dyn(wpe, PositionsFor(input, pastLen));

                var presents = new List<object>();
                var pasts = !(past is null)
                    ? tf.unstack(past, axis: 1)
                    : Enumerable.Repeat<object>(null, (int)hParams.n_layer);

                int layer = 0;
                foreach(dynamic existingPast in pasts)
                {
                    var block = Block(h, $"h{layer}", past: existingPast, hParams: hParams);
                    h = block.Item1;
                    presents.Add(block.Item2);
                    layer++;
                }

                result["present"] = tf.stack(presents.ToArray(), axis: 1);
                h = Norm(h, "ln_f");

                // Language model loss.  Do tokens <n predict token n?
                var hFlat = tf.reshape_dyn(h, new [] { sequence * batch, (int)hParams.n_embd });
                Tensor logits = tf.matmul(hFlat, wte, transpose_b: true);
                logits = tf.reshape_dyn(logits, new [] { batch, sequence, (int)hParams.n_vocab });
                result["logits"] = logits;
            }
            return result;
        }

        public static HParams LoadHParams(string modelName) {
            var hParams = DefaultHParams;
            string paramsOverridePath = Path.Combine("models", modelName, "hparams.json");
            var overrides = JsonConvert.DeserializeObject<Dictionary<string, object>>(File.ReadAllText(paramsOverridePath));
            var pyDict = new PythonDict<object, object>();
            foreach (var entry in overrides)
                pyDict.Add(entry.Key, entry.Value);
            hParams.override_from_dict(pyDict);
            return hParams;
        }
    }
}
