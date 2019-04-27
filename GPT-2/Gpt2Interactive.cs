namespace Gradient.Samples.GPT2
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using ManyConsole.CommandLineUtils;
    using Newtonsoft.Json;
    using numpy;
    using Python.Runtime;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.train;

    class Gpt2Interactive: ConsoleCommand
    {
        /// <summary>
        /// Interactively run the model
        /// </summary>
        /// <param name="modelName">Which model to use</param>
        /// <param name="checkpoint">Which checkpoint to load</param>
        /// <param name="seed">Seed for random number generators, fix seed to reproduce results</param>
        /// <param name="sampleCount">Number of samples to return total</param>
        /// <param name="batchSize">Number of batches (only affects speed/memory).  Must divide sampleCount.</param>
        /// <param name="length">Number of tokens in generated text, if null (default), is
        ///     determined by model hyperparameters</param>
        /// <param name="temperature">randomness in boltzmann distribution.
        ///     Lower temperature results in less random completions. As the 
        ///     temperature approaches zero, the model will become deterministic and
        ///     repetitive. Higher temperature results in more random completions.</param>
        /// <param name="topK">Controls diversity. 1 means only 1 word is
        ///     considered for each step (token), resulting in deterministic completions,
        ///     while 40 means 40 words are considered at each step. 0 (default) is a
        ///     special setting meaning no restrictions. 40 generally is a good value.
        /// </param>
        public static void Run(string modelName = "117M", string checkpoint = null, int? seed = null,
            int sampleCount = 1,
            int batchSize = 1, int? length = null, float temperature = 1, int topK = 0) {
            if (sampleCount % batchSize != 0)
                throw new ArgumentException();

            var encoder = Gpt2Encoder.LoadEncoder(modelName);
            var hParams = Gpt2Model.LoadHParams(modelName);

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
                checkpoint = checkpoint ?? tf.train.latest_checkpoint(Path.Combine("models", modelName));
                saver.restore(sess, checkpoint);

                while (true) {
                    string text;
                    do {
                        Console.Write("Model prompt >>> ");
                        text = Console.ReadLine();
                        if (string.IsNullOrEmpty(text))
                            Console.WriteLine("Prompt should not be empty");
                    } while (string.IsNullOrEmpty(text));

                    var contextTokens = encoder.Encode(text);
                    int generated = 0;
                    foreach (var _ in Enumerable.Range(0, sampleCount / batchSize)) {
                        var @out = sess.run(output, feed_dict: new PythonDict<object, object> {
                            [context] = Enumerable.Repeat(contextTokens, batchSize),
                        })[Range.All, Range.StartAt(contextTokens.Count)];
                        foreach (int i in Enumerable.Range(0, batchSize)) {
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

        public Gpt2Interactive() {
            this.IsCommand("run");
            this.HasOption("m|model=", "Which model to use", name => this.ModelName = name);
            this.HasOption("s|seed=",
                "Explicitly set seed for random generators to get reproducible results",
                (int s) => this.Seed = s);
            this.HasOption("c|sample-count=", "Number of samples to generate for each prompt",
                (int count) => this.SampleCount = count);
            this.HasOption("b|batch-size=", "Size of the batch, must divide sample-count",
                (int size) => this.BatchSize = size);
            this.HasOption("l|sample-length=", "Length of the generated samples",
                (int len) => this.Length = len);
            this.HasOption("t|temperature=", "Randomness of the generated text",
                (float t) => this.Temperature = t);
            this.HasOption("k|top-k=", "Number of words to consider for each step",
                (int k) => this.TopK = k);
            this.HasOption("r|run=", "For tuned models, which run to use",
                run => this.RunName = run);
            this.HasOption("checkpoint=", "Which run checkpoint to use (default: latest)",
                checkpoint => this.Checkpoint = checkpoint);
        }

        public string ModelName { get; set; } = "117M";
        public int? Seed { get; set; }
        public int SampleCount { get; set; } = 1;
        public int BatchSize { get; set; } = 1;
        public int? Length { get; set; }
        public float Temperature { get; set; } = 1;
        public int TopK { get; set; }
        public string RunName { get; set; }
        public string Checkpoint { get; set; } = "latest";

        public override int Run(string[] remainingArguments) {
            string checkpoint = Gpt2Checkpoints.ProcessCheckpointConfig(
                gpt2Root: Environment.CurrentDirectory,
                checkpoint: this.Checkpoint,
                modelName: this.ModelName,
                runName: this.RunName);

            Run(modelName: this.ModelName,
                checkpoint: checkpoint,
                seed: this.Seed,
                sampleCount: this.SampleCount,
                batchSize: this.BatchSize,
                length: this.Length,
                temperature: this.Temperature,
                topK: this.TopK);

            return 0;
        }

        static readonly string Delimiter = new string(Enumerable.Repeat('=', 40).ToArray());
    }
}
