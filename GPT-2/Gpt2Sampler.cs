namespace LostTech.Gradient.Samples.GPT2
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using tensorflow;
    using tensorflow.contrib.training;
    using tensorflow.python.framework.dtypes;
    using tensorflow.python.ops.variable_scope;

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
                Tensor minValues = values[Range.All, -1, tf.newaxis];
                return tf.where(logits < minValues,
                    tf.ones_like(logits, dtype: logits.dtype) * -1e10,
                    logits);
            }

            Tensor isTopKZero = tf.equal(topK, 0);
            return tf.cond(isTopKZero,
                true_fn: PythonFunctionContainer.Of(() => logits),
                false_fn: PythonFunctionContainer.Of(TopK));
        }

        public static Tensor SampleSequence(HParams hParams, int length,
            string startToken = null, int? batchSize = null, dynamic context = null,
            float temperature = 1, int topK = 0)
        {
            if (((startToken is null) ^ (context is null)) == false)
                throw new ArgumentException($"Exactly one of {nameof(startToken)} or {nameof(context)} has to be specified");

            SortedDictionary<string, dynamic> Step(HParams @params, Tensor tokens, dynamic past = null)
            {
                var lmOutput = Gpt2Model.Model(hParams: @params, input: tokens, past: past, reuse: _ReuseMode.AUTO_REUSE);

                var logits = lmOutput["logits"][Range.All, Range.All, Range.EndAt((int)@params.get("n_vocab"))];
                Tensor presents = lmOutput["present"];
                int?[] pastShape = Gpt2Model.PastShape(hParams: @params, batchSize: batchSize);
                presents.set_shape_(pastShape.Cast<object>());

                return new SortedDictionary<string, object>
                {
                    ["logits"] = logits,
                    ["presents"] = presents,
                };
            }

            Tensor result = null;
            new name_scope("sample_sequence").Use(_ =>
            {
                // Don't feed the last context token -- leave that to the loop below
                // TODO: Would be slightly faster if we called step on the entire context,
                // rather than leaving the last token transformer calculation to the while loop.
                var contextOutput = Step(hParams, context[Range.All, Range.EndAt(new Index(1, fromEnd: true))]);

                Tensor[] Body(object past, dynamic prev, object output)
                {
                    var nextOutputs = Step(hParams, prev[Range.All, tf.newaxis], past: past);
                    Tensor logits = nextOutputs["logits"][Range.All, -1, Range.All] / tf.constant(temperature, dtypes.float32_ref);
                    logits = TopLogits(logits, topK: topK);
                    var samples = tf.multinomial(logits, num_samples: 1, output_dtype: tf.int32);
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
                    context[Range.All, -1],
                    context,
                };
                TensorShape[] shapeInvariants = new[]{
                    new TensorShape(Gpt2Model.PastShape(hParams: hParams, batchSize: batchSize)),
                    new TensorShape(batchSize),
                    new TensorShape((int?)batchSize, (int?)null),
                };
                result = tf.while_loop(
                    cond: PythonFunctionContainer.Of<object, object, object, bool>(True),
                    body: PythonFunctionContainer.Of(new Func<object, object, object, Tensor[]>(Body)),
                    parallel_iterations: 10,
                    swap_memory: false,
                    name: null,
                    maximum_iterations: tf.constant(length),
                    loop_vars: loopVars,
                    shape_invariants: shapeInvariants,
                    back_prop: false)
                    [2];
            });
            return result;
        }
    }
}
