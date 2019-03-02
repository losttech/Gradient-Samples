namespace Gradient.Samples.GPT2
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Newtonsoft.Json;
    using numpy;
    using Python.Runtime;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.train;

    static class Gpt2Interactive
    {
        /// <summary>
        /// Interactively run the model
        /// </summary>
        /// <param name="modelName">Which model to use</param>
        /// <param name="seed">Seed for random number generators, fix seed to reproduce results</param>
        /// <param name="sampleCount">Number of samples to return total</param>
        /// <param name="batchSize">Number of batches (only affects speed/memory).  Must divide sampleCount.</param>
        /// <param name="length">Number of tokens in generated text, if null (default), is
        /// determined by model hyperparameters</param>
        /// <param name="temperature">randomness in boltzmann distribution.
        /// Lower temperature results in less random completions. As the 
        /// temperature approaches zero, the model will become deterministic and
        /// repetitive. Higher temperature results in more random completions.</param>
        /// <param name="topK">Controls diversity. 1 means only 1 word is
        /// considered for each step (token), resulting in deterministic completions,
        /// while 40 means 40 words are considered at each step. 0 (default) is a
        /// special setting meaning no restrictions. 40 generally is a good value.
        /// </param>
        public static void Run(string modelName = "117M", int? seed = null, int sampleCount = 1,
            int batchSize = 1, int? length = null, float temperature = 1, int topK = 0)
        {
            if (sampleCount % batchSize != 0)
                throw new ArgumentException();

            var encoder = Gpt2Encoder.LoadEncoder(modelName);
            var hParams = Gpt2Model.DefaultHParams;
            string paramsOverridePath = Path.Combine("models", modelName, "hparams.json");
            var overrides = JsonConvert.DeserializeObject<Dictionary<string, object>>(File.ReadAllText(paramsOverridePath));
            var pyDict = new PythonDict<object, object>();
            foreach (var entry in overrides)
                pyDict.Add(entry.Key, entry.Value);
            hParams.override_from_dict(pyDict);

            int nCtx = ((dynamic)hParams).n_ctx;
            if (length is null)
                length = nCtx;
            else if (length > nCtx)
                throw new ArgumentException("Can't get samples longer than window size: " + hParams.get("n_ctx"));

            new Session(graph: new Graph()).UseSelf(sess => {
                var context = tf.placeholder(tf.int32, new TensorShape(batchSize, null));
                tf.set_random_seed(seed);

                var output = Gpt2Sampler.SampleSequence(
                    hParams: hParams,
                    length: length.Value,
                    context: context,
                    batchSize: batchSize,
                    temperature: temperature,
                    topK: topK);

                var saver = new Saver();
                var checkpoint = tf.train.latest_checkpoint(Path.Combine("models", modelName));
                saver.restore(sess, checkpoint);

                while (true)
                {
                    string text;
                    do
                    {
                        Console.Write("Model prompt >>> ");
                        text = Console.ReadLine();
                        if (string.IsNullOrEmpty(text))
                            Console.WriteLine("Prompt should not be empty");
                    }  while(string.IsNullOrEmpty(text));

                    var contextTokens = encoder.Encode(text);
                    int generated = 0;
                    foreach(var _ in Enumerable.Range(0, sampleCount / batchSize))
                    {
                        var @out = sess.run(output, feed_dict: new PythonDict<object, object>
                        {
                            [context] = Enumerable.Repeat(contextTokens, batchSize),
                        }).__getitem__(ValueTuple.Create(Range.All, Range.StartAt(contextTokens.Count)));
                        foreach(int i in Enumerable.Range(0, batchSize))
                        {
                            generated++;
                            ndarray part = @out[i];
                            text = encoder.Decode(part);
                            Console.WriteLine($"{Delimiter} SAMPLE {generated} {Delimiter}");
                            Console.WriteLine(text);
                        }
                    }
                    Console.Write(Delimiter);
                    Console.WriteLine(Delimiter);
                }
            });
        }

        static readonly string Delimiter = new string(Enumerable.Repeat('=', 40).ToArray());
    }
}
