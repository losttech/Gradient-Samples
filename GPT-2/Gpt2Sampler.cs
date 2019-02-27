namespace Gradient.Samples.GPT2
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using numpy;
    using tensorflow;
    using tensorflow.contrib.training;
    using tensorflow.python.ops.variable_scope;

    static class Gpt2Sampler
    {
        static Tensor TopLogits(Tensor logits, int topK)
        {
            if (topK == 0)
                // no truncation
                return logits;

            Tensor TopK()
            {
                var valuesIndices = tf.nn.top_k_dyn(logits, k: topK);
                var values = valuesIndices.Item1;
                var minValues = values[Range.All(), -1, tf.newaxis];
                return tf.where_dyn(logits < minValues,
                    tf.ones_like(logits, dtype: logits.dtype) * -1e10,
                    logits);
            }

            return tf.cond(tf.equal(topK, 0),
                PythonFunctionContainer.Of(() => logits),
                PythonFunctionContainer.Of(TopK));
        }

        public static Tensor SampleSequence(HParams hParams, int length,
            string startToken = null, int? batchSize = null, dynamic context = null,
            float temperature = 1, int topK = 0)
        {
            if (((startToken == null) ^ (context == null)) == false)
                throw new ArgumentException($"Exactly one of {nameof(startToken)} or {nameof(context)} has to be specified");

            SortedDictionary<string, dynamic> Step(HParams @params, Tensor tokens, dynamic past = null)
            {
                var lmOutput = Gpt2Model.Model(hParams: @params, input: tokens, past: past, reuse: _ReuseMode.AUTO_REUSE);

                var logits = lmOutput["logits"][Range.All(), Range.All(), Range.ToEnd(@params.get("n_vocab"))];
                Tensor presents = lmOutput["present"];
                presents.set_shape_(Gpt2Model.PastShape(hParams: @params, batchSize: batchSize).Cast<object>());

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
                var contextOutput = Step(hParams, context[Range.All(), Range.ToEnd(new Index(1, fromEnd: true))]);

                Tensor[] Body(object past, dynamic prev, object output)
                {
                    var nextOutputs = Step(hParams, prev[Range.All(), tf.newaxis], past: past);
                    var logits = nextOutputs["logits"][Range.All(), -1, Range.All()] / tf.to_float(temperature);
                    logits = TopLogits(logits, topK: topK);
                    var samples = tf.multinomial(logits, num_samples: 1, output_dtype: tf.int32);
                    return new Tensor[]
                    {
                        tf.concat(new []{ past, nextOutputs["presents"]}, axis: -2),
                        tf.squeeze(samples, axis: new[]{1}),
                        tf.concat(new []{output, samples}, axis: 1),
                    };
                }

                bool True(object _unused) => true;

                result = tf.while_loop(cond: PythonFunctionContainer.Of<object, bool>(True),
                    body: PythonFunctionContainer.Of(new Func<object, object, object, Tensor[]>(Body)),
                    maximum_iterations: np.array(1),
                    loop_vars: new[]{
                        contextOutput["presents"],
                        context[Range.All(), -1],
                        context
                    },
                    shape_invariants: new[]{
                        new TensorShape(Gpt2Model.PastShape(hParams: hParams, batchSize: batchSize)),
                        new TensorShape(batchSize),
                        new TensorShape(batchSize, null),
                    },
                    back_prop: false)
                    [2];
            });
            return result;
        }
    }
}
