﻿// ported from https://github.com/nshepperd/gpt-2

namespace LostTech.Gradient.Samples.GPT2 {
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Threading;
    using LostTech.Gradient.BuiltIns;
    using numpy;

    using tensorflow;
    using tensorflow.contrib.training;
    using tensorflow.train;

    using static System.FormattableString;

    using DataSet = System.Collections.Generic.List<numpy.ndarray>;

    public class Gpt2Trainer {
        const string SampleDir = "samples";

        readonly DataSet dataset;
        readonly Gpt2Encoder encoder;
        readonly HParams hParams;
        readonly int batchSize;
        readonly int sampleLength;
        readonly Random random;

        public Gpt2Trainer(DataSet dataset, Gpt2Encoder encoder, HParams hParams,
            int batchSize, int sampleLength, Random random) {
            this.dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
            this.encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
            this.hParams = hParams ?? throw new ArgumentNullException(nameof(hParams));
            this.batchSize = batchSize;
            this.sampleLength = sampleLength;
            this.random = random ?? throw new ArgumentNullException(nameof(random));
        }

        public int SaveEvery { get; set; } = 1000;
        public int SampleEvery { get; set; } = 100;
        public int SampleNum { get; set; } = 1;

        public void Train(string checkpoint, string run, int? counter, dynamic sessionConfig = null, CancellationToken cancellation = default) {
            Session session = sessionConfig is null
                ? Session.NewDyn(config: sessionConfig)
                : new Session();
            using (session.StartUsing()) {
                var context = tf.placeholder(tf.int32, new TensorShape(this.batchSize, null));
                var output = Gpt2Model.Model(this.hParams, input: context);
                Tensor labels = context[.., 1..];
                Tensor logits = output["logits"][.., ..^1];
                var loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits_dyn(
                        labels: labels,
                        logits: logits));

                var sample = Gpt2Sampler.SampleSequence(
                    this.hParams,
                    length: this.sampleLength,
                    context: context,
                    batchSize: this.batchSize,
                    temperature: 1.0f,
                    topK: 40);

                var trainVars = ((PythonList<Variable>)tf.trainable_variables()).Where(var => var.name.Contains("model")).ToPyList();
                var optimizer = new AdamOptimizer(learning_rate: 0.0002).minimize(loss, var_list: trainVars);

                var saver = new Saver(
                    var_list: trainVars,
                    max_to_keep: 5,
                    keep_checkpoint_every_n_hours: 1);

                session.run(tf.global_variables_initializer());

                Console.WriteLine("Loading checkpoint " + checkpoint);
                saver.restore(session, checkpoint);

                Console.WriteLine("Loading dataset...");
                var sampler = new TrainingSampler(this.dataset, this.random);
                Console.WriteLine($"Dataset has {sampler.TokenCount} tokens");

                string counterFile = Path.Combine(Gpt2Checkpoints.CheckpointDir, run, "counter");
                if (counter is null && File.Exists(counterFile))
                    counter = int.Parse(File.ReadAllText(counterFile), CultureInfo.InvariantCulture) + 1;
                counter = counter ?? 1;

                string runCheckpointDir = Path.Combine(Gpt2Checkpoints.CheckpointDir, run);
                string runSampleDir = Path.Combine(SampleDir, run);

                void Save() {
                    Directory.CreateDirectory(runCheckpointDir);
                    Console.WriteLine("Saving " + Path.Combine(runCheckpointDir, Invariant($"model-{counter}")));
                    saver.save(session,
                        Path.Combine(runCheckpointDir, "model"),
                        global_step: counter.Value);
                    File.WriteAllText(path: counterFile, contents: Invariant($"{counter}"));
                }

                void GenerateSamples() {
                    var contextTokens = np.array(new[] { this.encoder.EncodedEndOfText });
                    var allText = new List<string>();
                    int index = 0;
                    string text = null;
                    while (index < this.SampleNum) {
                        ndarray<int> @out = session.run(sample, feed_dict: new Dictionary<object, object> {
                            [context] = Enumerable.Repeat(contextTokens, this.batchSize),
                        });
                        foreach (int i in Enumerable.Range(0, Math.Min(this.SampleNum - index, this.batchSize))) {
                            text = this.encoder.Decode((ndarray<int>)@out[i]);
                            text = Invariant($"======== SAMPLE {index + 1} ========\n{text}\n");
                            allText.Add(text);
                            index++;
                        }
                    }
                    Console.WriteLine(text);
                    Directory.CreateDirectory(runSampleDir);
                    File.WriteAllLines(
                        path: Path.Combine(runSampleDir, Invariant($"samples-{counter}")),
                        contents: allText);
                }

                var avgLoss = (0.0, 0.0);
                var startTime = DateTime.Now;

                while (!cancellation.IsCancellationRequested) {
                    if (counter % this.SaveEvery == 0)
                        Save();
                    if (counter % this.SampleEvery == 0)
                        GenerateSamples();

                    var batch = Enumerable.Range(0, this.batchSize)
                        .Select(_ => sampler.Sample(1024))
                        .ToArray();

                    var placeholderValues = new Dictionary<object, object> {
                        [context] = batch,
                    };
                    var tuple = session.run((optimizer, loss), feed_dict: placeholderValues);

                    var lv = tuple.Item2;

                    avgLoss = (avgLoss.Item1 * 0.99 + lv, avgLoss.Item2 * 0.99 + 1);

                    Console.WriteLine($"[{counter} | {DateTime.Now-startTime}] loss={lv} avg={avgLoss.Item1/avgLoss.Item2}");

                    counter++;
                }

                Console.WriteLine("Interrupted");
                Save();
            }
        }

        class TrainingSampler {
            readonly DataSet chunks;
            readonly List<int> boundaries = new List<int> { 0 };
            readonly Random random;
            public int TokenCount { get; }

            public TrainingSampler(DataSet chunks, Random random) {
                this.random = random ?? throw new ArgumentNullException(nameof(random));
                this.chunks = chunks ?? throw new ArgumentNullException(nameof(chunks));
                this.TokenCount = chunks.Sum(chunk => chunk.shape.Item1);
                if (this.TokenCount == 0)
                    throw new ArgumentException("Dataset is empty", paramName: nameof(chunks));

                foreach(var chunk in chunks)
                    this.boundaries.Add(this.boundaries[this.boundaries.Count - 1] + chunk.shape.Item1);
            }

            public ndarray Sample(int length) {
                if (length >= this.TokenCount / this.chunks.Count)
                    throw new ArgumentException($"Dataset files are too small to sample {length} tokens at a time." +
                        $"Maximum is {this.TokenCount/this.chunks.Count}.");

                while (true) {
                    int index = this.random.Next(this.TokenCount - length);
                    int i = Algo.BinarySearch(j => this.boundaries[j] > index,
                        lo: 0, hi: this.boundaries.Count - 1) - 1;

                    if (this.boundaries[i+1] > index + length) {
                        int withinChunk = index - this.boundaries[i];
                        dynamic chunk = this.chunks[i];
                        return chunk[withinChunk .. (withinChunk + length)];
                    }
                }
            }
        }
    }
}
