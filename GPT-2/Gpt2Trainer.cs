// ported from https://github.com/nshepperd/gpt-2

namespace Gradient.Samples.GPT2 {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Threading;

    using numpy;

    using Python.Runtime;

    using SharPy.Runtime;

    using tensorflow;
    using tensorflow.contrib.training;
    using tensorflow.train;

    using static System.FormattableString;

    using DataSet = System.Collections.Generic.List<numpy.ndarray>;

    public class Gpt2Trainer {
        const string CheckpointDir = "checkpoint";
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

        public static string GetLatestCheckpoint(string gpt2Root, string modelName, string run)
            => tf.train.latest_checkpoint(Path.Combine(gpt2Root, CheckpointDir, run))
            ?? GetOriginalCheckpoint(gpt2Root, modelName);

        public static string GetOriginalCheckpoint(string gpt2Root, string modelName)
            => tf.train.latest_checkpoint(Path.Combine(gpt2Root, "models", modelName));

        public void Train(string checkpoint, string run, CancellationToken cancellation) {
            new Session().UseSelf(session => {
                var context = tf.placeholder(tf.int32, new int?[] { this.batchSize, null }.Cast<object>());
                var output = Gpt2Model.Model(this.hParams, input: context);
                Tensor labels = context[Range.All, Range.StartAt(1)];
                Tensor logits = output["logits"][Range.All, Range.EndAt(new Index(1, fromEnd: true))];
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

                var trainVars = tf.trainable_variables().Where((dynamic var) => var.name.Contains("model"));
                var optimizer = new AdamOptimizer().minimize(loss, var_list: trainVars);

                var saver = new Saver(
                    var_list: trainVars,
                    max_to_keep: 5,
                    keep_checkpoint_every_n_hours: 1);

                session.run(tf.global_variables_initializer());

                Debug.WriteLine("Loading checkpoint " + checkpoint);
                saver.restore(session, checkpoint);

                Debug.WriteLine("Loading dataset...");
                var sampler = new TrainingSampler(this.dataset, this.random);
                Debug.WriteLine($"Dataset has {sampler.TokenCount} tokens");

                int counter = 1;
                string counterFile = Path.Combine(CheckpointDir, run, "counter");
                if (File.Exists(counterFile))
                    counter = int.Parse(File.ReadAllText(counterFile), CultureInfo.InvariantCulture) + 1;

                string runCheckpointDir = Path.Combine(CheckpointDir, run);
                string runSampleDir = Path.Combine(SampleDir, run);

                void Save() {
                    Directory.CreateDirectory(runCheckpointDir);
                    Debug.WriteLine("Saving " + Path.Combine(runCheckpointDir, Invariant($"model-{counter}")));
                    saver.save(session,
                        Path.Combine(runCheckpointDir, "model"),
                        global_step: counter);
                    File.WriteAllText(path: counterFile, contents: Invariant($"{counter}"));
                }

                void GenerateSamples() {
                    var contextTokens = np.array(new[] { this.encoder.EncodedEndOfText });
                    var allText = new List<string>();
                    int index = 0;
                    string text = null;
                    while (index < this.SampleNum) {
                        var @out = session.run(sample, feed_dict: new PythonDict<object, object> {
                            [context] = Enumerable.Repeat(contextTokens, this.batchSize),
                        });
                        foreach (int i in Enumerable.Range(0, Math.Min(this.SampleNum - index, this.batchSize))) {
                            text = this.encoder.Decode(@out[i]);
                            text = Invariant($"======== SAMPLE {index + 1} ========\n{text}\n");
                            allText.Add(text);
                            index++;
                        }
                    }
                    Debug.WriteLine(text);
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

                    var placeholderValues = new PythonDict<object, object> {
                        [context] = batch.ToPythonList(),
                    };
                    var tuple = session.run_dyn((optimizer, loss), feed_dict: placeholderValues);

                    var lv = tuple.Item2;

                    avgLoss = (avgLoss.Item1 * 0.99 + lv, avgLoss.Item2 * 0.99 + 1);

                    Debug.WriteLine($"[{counter} | {DateTime.Now-startTime}] loss={lv} avg={avgLoss.Item1/avgLoss.Item2}");

                    counter++;
                }

                Debug.WriteLine("Interrupted");
                Save();
            });
        }

        internal static DataSet LoadDataset(Gpt2Encoder encoder, string path, string pattern = "*") {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));
            var paths = new List<string>();
            if (Directory.Exists(path))
                paths.AddRange(Directory.EnumerateFiles(path, searchPattern: pattern, SearchOption.AllDirectories));
            else
                paths.Add(path);

            return LoadDataset(encoder, paths);
        }

        internal static DataSet LoadDataset(Gpt2Encoder encoder, List<string> fileNames) {
            if (encoder == null) throw new ArgumentNullException(nameof(encoder));

            var tokenChunks = new DataSet();
            foreach (string file in fileNames) {
                Debug.WriteLine($"Reading {file}");
                if (Path.GetExtension(file) == ".npz") {
                    // pre-encoded
                    dynamic npzObject = np.load(file);
                    var npz = npzObject.__enter__();
                    foreach (var item in npz.files)
                        tokenChunks.Add(npz[item]);
                    npzObject.__exit__();
                } else {
                    string rawText = File.ReadAllText(file);
                    if (string.IsNullOrWhiteSpace(rawText))
                        continue;
                    dynamic numpy = Py.Import("numpy");
                    PyObject tokens = numpy.stack(encoder.Encode(rawText));
                    tokenChunks.Add(ndarray.Wrap(tokens));
                }
            }

            return tokenChunks;
        }

        static int BinarySearch(Func<int, bool> predicate, int lo, int hi) {
            if (predicate == null)
                throw new ArgumentNullException(nameof(predicate));
            if (predicate(lo) || !predicate(hi))
                throw new ArgumentException();
            while(hi > lo + 1) {
                int mid = (lo + hi) / 2;
                if (predicate(mid))
                    hi = mid;
                else
                    lo = mid;
            }
            return hi;
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
                    int i = BinarySearch(j => this.boundaries[j] > index,
                        lo: 0, hi: this.boundaries.Count - 1) - 1;

                    if (this.boundaries[i+1] > index + length) {
                        int withinChunk = index - this.boundaries[i];
                        dynamic chunk = this.chunks[i];
                        return chunk[new Range(withinChunk, withinChunk + length - 1)];
                    }
                }
            }
        }

        public static string ProcessCheckpointConfig(string gpt2Root, string checkpoint,
            string modelName, string runName) {
            switch (checkpoint) {
            case "latest":
                checkpoint = GetLatestCheckpoint(gpt2Root, modelName, runName);
                break;
            case "fresh":
                checkpoint = GetOriginalCheckpoint(gpt2Root, modelName);
                break;
            }

            return checkpoint;
        }
    }
}
