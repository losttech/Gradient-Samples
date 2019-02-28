namespace Gradient.Samples.GPT2
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using numpy;
    using Python.Runtime;
    using tensorflow;
    using tensorflow.contrib.training;
    using tensorflow.python.ops.variable_scope;

    using Range = RangeWorkaround;

    static class Gpt2Sampler
    {
        static readonly dynamic AUTO_REUSE = Py.Import("tensorflow").GetAttr("AUTO_REUSE");
        static Tensor TopLogits(Tensor logits, int topK)
        {
            if (topK == 0)
                // no truncation
                return logits;

            Tensor TopK()
            {
                var valuesIndices = tf.nn.top_k_dyn(logits, k: topK);
                var values = valuesIndices.Item1;
                var minValues = values.__getitem__(ValueTuple.Create(Range.All(), -1, tf.newaxis));
                return tf.where_dyn(logits < minValues,
                    tf.ones_like(logits, dtype: logits.dtype) * -1e10,
                    logits);
            }

            return tf.cond(tf.equal(topK, 0),
                PythonFunctionContainer.Of(() => logits),
                PythonFunctionContainer.Of(TopK));
        }

        public static Tensor SampleSequence(dynamic hParams, int length,
            string startToken = null, int? batchSize = null, dynamic context = null,
            float temperature = 1, int topK = 0)
        {
            if (((startToken == null) ^ (context == null)) == false)
                throw new ArgumentException($"Exactly one of {nameof(startToken)} or {nameof(context)} has to be specified");

            SortedDictionary<string, dynamic> Step(dynamic @params, Tensor tokens, dynamic past = null)
            {
                var lmOutput = GPT2.Gpt2Model.Model(hParams: @params, input: tokens, past: past, reuse: AUTO_REUSE);

                var logits = lmOutput["logits"].__getitem__(ValueTuple.Create(Range.All(), Range.All(), Range.ToEnd((int)@params.n_vocab)));
                Tensor presents = lmOutput["present"];
                int?[] pastShape = GPT2.Gpt2Model.PastShape(hParams: @params, batchSize: batchSize);
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
                var contextOutput = Step(hParams, context.__getitem__(ValueTuple.Create(Range.All(), Range.ToEnd(new Index(1, fromEnd: true)))));

                Tensor[] Body(object past, dynamic prev, object output)
                {
                    var nextOutputs = Step(hParams, prev.__getitem__(ValueTuple.Create(Range.All(), tf.newaxis)), past: past);
                    Tensor logits = nextOutputs["logits"].__getitem__(ValueTuple.Create(Range.All(), -1, Range.All())) / tensorflow.tf.to_float(temperature);
                    logits = TopLogits(logits, topK: topK);
                    var samples = tf.multinomial_dyn(logits, num_samples: 1, output_dtype: tf.int32);
                    return new Tensor[]
                    {
                        tf.concat(new []{ past, nextOutputs["presents"]}, axis: -2),
                        tensorflow.tf.squeeze(samples, axis: new[]{1}),
                        tf.concat(new []{ output, samples}, axis: 1),
                    };
                }

                bool True(object _a, object _b, object _c) => true;

                dynamic[] loopVars = new[]{
                    contextOutput["presents"],
                    context.__getitem__(ValueTuple.Create(Range.All(), -1)),
                    context,
                };
                TensorShape[] shapeInvariants = new[]{
                    new TensorShape(GPT2.Gpt2Model.PastShape(hParams: hParams, batchSize: batchSize)),
                    new TensorShape(batchSize),
                    new TensorShape((int?)batchSize, (int?)null),
                };
                result = tensorflow.tf.while_loop(cond: PythonFunctionContainer.Of<object, object, object, bool>(True),
                    body: PythonFunctionContainer.Of(new Func<object, object, object, Tensor[]>(Body)),
                    maximum_iterations: np.array(length),
                    loop_vars: loopVars,
                    shape_invariants: shapeInvariants,
                    back_prop: false)
                    [2];
            });
            return result;
        }
    }
}
