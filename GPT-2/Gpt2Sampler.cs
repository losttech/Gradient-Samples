namespace LostTech.Gradient.Samples.GPT2
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using tensorflow;
    using tensorflow.compat.v1;
    using tensorflow.python.framework.dtypes;
    using tensorflow.python.ops.variable_scope;
    using name_scope = tensorflow.name_scope;

    public static class Gpt2Sampler
    {
        static Tensor TopLogits(Tensor logits, int topK)
        {
            if (topK == 0)
                // no truncation
                return logits;

            Tensor TopK()
            {
                var valuesIndices = tf.nn.top_k(logits, k: topK);
                var values = valuesIndices[0];
                Tensor minValues = values[.., ^1, tf.newaxis];
                return tf.where(logits < minValues,
                    tf.ones_like(logits, dtype: logits.dtype) * -1e10,
                    logits);
            }

            Tensor isTopKZero = tf.equal(topK, 0);
            return tf.cond(isTopKZero,
                true_fn: PythonFunctionContainer.Of(() => logits),
                false_fn: PythonFunctionContainer.Of(TopK));
        }

        public static Tensor SampleSequence(IReadOnlyDictionary<string, int> hParams, int length,
            string startToken = null, int? batchSize = null, dynamic context = null,
            float temperature = 1, int topK = 0)
        {
            if (((startToken is null) ^ (context is null)) == false)
                throw new ArgumentException($"Exactly one of {nameof(startToken)} or {nameof(context)} has to be specified");

            SortedDictionary<string, dynamic> Step(IReadOnlyDictionary<string, int> @params, Tensor tokens, dynamic past = null)
            {
                var lmOutput = Gpt2Model.Model(hParams: @params, input: tokens, past: past, reuse: _ReuseMode.AUTO_REUSE);

                var logits = lmOutput["logits"][.., .., ..@params["n_vocab"]];
                Tensor presents = lmOutput["present"];
                int?[] pastShape = Gpt2Model.PastShape(hParams: @params, batchSize: batchSize);
                presents.set_shape_(new TensorShape(pastShape));

                return new SortedDictionary<string, object>
                {
                    ["logits"] = logits,
                    ["presents"] = presents,
                };
            }

            Tensor result = null;
            using (new name_scope("sample_sequence").StartUsing())
            {
                // Don't feed the last context token -- leave that to the loop below
                // TODO: Would be slightly faster if we called step on the entire context,
                // rather than leaving the last token transformer calculation to the while loop.
                var contextOutput = Step(hParams, context[.., ..^1]);

                Tensor[] Body(object past, dynamic prev, object output)
                {
                    var nextOutputs = Step(hParams, prev[.., tf.newaxis], past: past);
                    Tensor logits = nextOutputs["logits"][.., ^1, ..] / tf.constant(temperature, dtypes.float32_ref);
                    logits = TopLogits(logits, topK: topK);
                    var samples = v1.multinomial(logits, num_samples: 1, output_dtype: tf.int32);
                    return new Tensor[]
                    {
                        tf.concat(new []{ past, nextOutputs["presents"]}, axis: -2),
                        tf.squeeze(samples, axis: new[]{1}),
                        tf.concat(new []{ output, samples}, axis: 1),
                    };
                }

                bool True(object _a, object _b, object _c) => true;

                dynamic[] loopVars = new[]{
                    contextOutput["presents"],
                    context[.., ^1],
                    context,
                };
                TensorShape[] shapeInvariants = new[]{
                    new TensorShape(Gpt2Model.PastShape(hParams: hParams, batchSize: batchSize)),
                    new TensorShape(batchSize),
                    new TensorShape(batchSize, null),
                };
                Tensor maxTokens = tf.constant(length);
                // for some reason on CPU you can't sample longer texts
                // https://github.com/losttech/Gradient-Samples/issues/1
                if (!tf.test.is_gpu_available())
                    maxTokens -= tf.shape(context)[1];
                result = tf.while_loop_dyn(
                    cond: PythonFunctionContainer.Of<object, object, object, bool>(True),
                    body: PythonFunctionContainer.Of(new Func<object, object, object, Tensor[]>(Body)),
                    parallel_iterations: 10,
                    swap_memory: false,
                    name: null,
                    maximum_iterations: maxTokens,
                    loop_vars: loopVars,
                    shape_invariants: shapeInvariants,
                    back_prop: false)
                    [2];
            }
            return result;
        }
    }
}
