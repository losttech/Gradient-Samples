namespace Gradient.Samples.GPT2
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.contrib.training;

    static class Gpt2Model
    {
        static HParams DefaultHParams => new HParams(kwargs: new PythonDict<string, object> {
            ["n_vocab"] = 0,
            ["n_ctx"] = 1024,
            ["n_embd"] = 768,
            ["n_head"] = 12,
            ["n_layer"] = 12,
        });

        /// <summary>
        /// Deal with dynamic shape in tensorflow cleanly.
        /// </summary>
        static int[] ShapeList(ITensor tensor)
        {
            var @static = tensor.shape.as_list();
            dynamic dynamic = tf.shape(tensor);
            return @static.Select((size, index) => size ?? (int)dynamic[index]).ToArray();
        }

        static Tensor Softmax(Tensor input, int axis = -1)
        {
            var negative = input - tf.reduce_max(input, axis: axis, keepdims: true);
            var exp = tf.exp(negative);
            return exp / tf.reduce_sum(exp, axis: axis, keepdims: true);
        }

        static Tensor GeLU(Tensor input) =>
            (dynamic)input * 0.5 * (1 + tf.tanh(Math.Sqrt(2 / Math.PI) * (input + 0.044715 * tf.pow(input, 3))));

        /// <summary>
        /// Normalize to mean = 0, std = 1, then do a diagonal affine transform.
        /// </summary>
        static Tensor Norm(Tensor input, object scope, int axis = -1, double epsilon = 1e-5)
        {
            Tensor result = null;
            new variable_scope(scope).Use(_ =>
            {
                var nState = input.shape[-1];
                var g = tf.get_variable("g", nState, initializer: new constant_initializer(1));
                var b = tf.get_variable("b", nState, initializer: new constant_initializer(0));
                var mean = tf.reduce_mean(input, axis: axis, keepdims: true);
                var s = tf.reduce_mean(tf.square(input - mean), axis: axis, keepdims: true);
                result = (input - mean) * tf.rsqrt(s + epsilon);
                result = result * g + b;
            });
            return result;
        }

        /// <summary>
        /// Reshape the last dimension of input into [n, input.shape[-1]/n]
        /// </summary>
        static Tensor SplitStates(Tensor input, int n)
        {
            var shape = ShapeList(input).ToList();
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
            var shape = ShapeList(input).ToList();
            shape[shape.Count - 2] = shape[shape.Count - 2] * shape[shape.Count - 1];
            shape.RemoveAt(shape.Count - 1);
            return tf.reshape_dyn(input, shape);
        }

        static Tensor Conv1D(Tensor input, object scope, int nf, double wInitialStDev = 0.02)
        {
            Tensor result = null;
            new variable_scope(scope).Use(_ =>
            {
                int[] shape = ShapeList(input);
                var start = shape.Take(shape.Length - 1);
                int nx = shape.Last();
                var w = tf.get_variable("w", new TensorShape(1, nx, nf), initializer: new random_normal_initializer(stddev: wInitialStDev));
                var b = tf.get_variable("b", new TensorShape(nf), initializer: new constant_initializer(0));
                result = tf.reshape(
                    tf.matmul(
                        tf.reshape(input, new[] { -1, nx }),
                        tf.reshape(w, new[] { -1, nf })) + b,
                    start.Append(nf));
            });
            return result;
        }

        /// <summary>
        /// 1's in the lower triangle, counting from the lower right corner.
        /// Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        /// </summary>
        static Tensor AttentionMask(dynamic nd, dynamic ns, DType dtype = null)
        {
            var i = tf.range(nd)[Range.All(), null];
            var j = tf.range(ns);
            var m = i >= j - ns + nd;
            return tf.cast(m, dtype);
        }

        static ValueTuple<Tensor, object> Attention(Tensor input, object scope, int nState, Tensor past = null, dynamic hParams = null)
        {
            Trace.Assert(input.shape.ndims == 3);
            Trace.Assert(nState % (int)hParams.n_head == 0);
            if (past != null)
                Trace.Assert(past.shape.ndims == 5);

            Tensor SplitHeads(Tensor x) =>
                // From [batch, sequence, features] to [batch, heads, sequence, features]
                tf.transpose(SplitStates(x, hParams.h_head), new[] { 0, 2, 1, 3 });

            Tensor MergeHeads(Tensor x) =>
                // Reverse of split_heads
                MergeStates(tf.transpose(x, new[] { 0, 2, 1, 3 }));

            Tensor MaskAttentionWeights(Tensor w)
            {
                // w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
                var shape = ShapeList(w);
                int nd = shape[shape.Length - 2];
                int ns = shape[shape.Length - 1];
                var b = AttentionMask(nd, ns, dtype: w.dtype);
                b = tf.reshape(b, new[] { 1, 1, nd, ns });
                w = (dynamic)w * b - tf.cast(1e10, w.dtype) * (1 - (dynamic)b);
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
            dynamic present = null;
            new variable_scope(scope).Use(_ =>
            {
                var c = Conv1D(input, "c_attn", nState * 3);
                var qkv = ((IEnumerable)tf.split(c, 3, axis: 2)).Cast<Tensor>().Select(SplitHeads).ToArray();
                var q = qkv[0];
                var k = qkv[1];
                var v = qkv[2];

                present = tf.stack(new[] { k, v }, axis: 1);
                if (past != null)
                {
                    var pastKV = tf.unstack(past, axis: 1);
                    k = tf.concat(new[] { pastKV[0], k }, axis: -2);
                    v = tf.concat(new[] { pastKV[1], v }, axis: -2);
                }

                attention = MultiHeadAttention(q, k, v);
                attention = MergeHeads(attention);
                attention = Conv1D(attention, "c_proj", nState);
            });

            return ValueTuple.Create(attention, present);
        }

        static Tensor MLP(Tensor input, string scope, int nState, dynamic hParams = null)
        {
            Tensor result = null;
            new variable_scope(scope).Use(_ =>
            {
                var nx = input.shape[-1].value;
                var h = GeLU(Conv1D(input, "c_fc", nState));
                result = Conv1D(h, "c_proj", nx);
            });
            return result;
        }

        static ValueTuple<Tensor, dynamic> Block(Tensor input, string scope, Tensor past = null, dynamic hParams = null)
        {
            Tensor result = null;
            dynamic present = null;
            new variable_scope(scope).Use(_ =>
            {
                var nx = input.shape[-1].value;
                var attentionPresent = Attention(Norm(input, "ln_1"), "attn", nx, past: past, hParams: hParams);
                Tensor attention = attentionPresent.Item1;
                present = attentionPresent.Item2;
                input += (dynamic)attention;
                var m = MLP(Norm(input, "ln_2"), "mlp", nx * 4, hParams: hParams);
                input += m;
                result = input;
            });
            return ValueTuple.Create(result, present);
        }

        static int?[] PastShape(dynamic hParams = null, int? batchSize = null, int? sequence = null)
        {
            return new int?[]
            {
                batchSize,
                hParams.n_layer,
                2,
                hParams.n_head,
                sequence,
                (int)hParams.n_embed / (int)hParams.n_head,
            };
        }

        /// <summary>
        /// "Add a new axis of given size.
        /// </summary>
        static Tensor ExpandTile(dynamic value, int size)
        {
            value = tf.convert_to_tensor(value, name: "value");
            int ndims = value.shape.ndims;
            return tf.tile(tf.expand_dims(value, axis: 0), new[] { size }.Concat(Enumerable.Repeat(1, ndims)));
        }

        static Tensor PositionsFor(dynamic tokens, dynamic pastLength)
        {
            int batchSize = tf.shape(tokens)[0];
            int nSteps = tf.shape(tokens)[1];
            return ExpandTile(pastLength + tf.range(nSteps), batchSize);
        }

        static Dictionary<string, Tensor> Model(dynamic hParams, Tensor input, dynamic past = null, string scope = "model", bool reuse = false)
        {
            var result = new Dictionary<string, Tensor>();
            new variable_scope(scope, reuse: reuse).Use(_ =>
            {
                int[] batchSeq = ShapeList(input);
                int batch = batchSeq[0];
                int sequence = batchSeq[1];

                var wpe = tf.get_variable("wpe", new TensorShape(hParams.n_ctx, hParams.n_embd), initializer: new random_normal_initializer(stddev: 0.01));
                var wte = tf.get_variable("wte", new TensorShape(hParams.n_vocab, hParams.n_embd), initializer: new random_normal_initializer(stddev: 0.02));

                int pastLen = past == null ? 0 : tf.shape(past)[-2];
                var h = tf.gather_dyn(wte, input) + tf.gather_dyn(wpe, PositionsFor(input, pastLen));

                var presents = new List<object>();
                var pasts = past != null
                    ? tf.unstack(past, axis: 1)
                    : Enumerable.Repeat<object>(null, hParams.n_layer);

                int layer = 0;
                foreach(dynamic existingPast in pasts)
                {
                    var block = Block(h, $"h{layer}", past: existingPast, hParams: hParams);
                    h = block.Item1;
                    presents.Add(block.Item2);
                    layer++;
                }

                result["present"] = tf.stack(presents, axis: 1);
                h = Norm(h, "ln_f");

                // Language model loss.  Do tokens <n predict token n?
                var hFlat = tf.reshape(h, new int[] { batch * sequence, hParams.n_embd });
                Tensor logits = tf.matmul(hFlat, wte, transpose_b: true);
                logits = tf.reshape(logits, new int[] { batch, sequence, hParams.n_vocab });
                result["logits"] = logits;
            });
            return result;
        }
    }
}
