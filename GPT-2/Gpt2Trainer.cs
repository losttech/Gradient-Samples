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
        readonly DataSet dataset;
        readonly HParams hParams;
        readonly int batchSize;
        readonly Random random;

        public Gpt2Trainer(DataSet dataset, HParams hParams, int batchSize, Random random) {
            this.dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
            this.hParams = hParams ?? throw new ArgumentNullException(nameof(hParams));
            this.batchSize = batchSize;
            this.random = random ?? throw new ArgumentNullException(nameof(random));
        }

        public int SaveEvery { get; set; } = 1000;

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
                string counterFile = GetCounterFileName(run);
                if (File.Exists(counterFile))
                    counter = int.Parse(File.ReadAllText(counterFile), CultureInfo.InvariantCulture) + 1;

                var avgLoss = (0.0, 0.0);
                var startTime = DateTime.Now;

                while (!cancellation.IsCancellationRequested) {
                    this.BeforeEpoch?.Invoke(counter, run, session, context);
                    if (counter % this.SaveEvery == 0)
                        Save(run, counter, session, saver);

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
                Save(run, counter, session, saver);
            });
        }

        public delegate void EpochHandler(int epoch, string run, Session session, dynamic context);
        public event EpochHandler BeforeEpoch;

        static string GetCounterFileName(string run)
            => Path.Combine(Gpt2Checkpoints.CheckpointDir, run, "counter");

        static void Save(string run, int counter, Session session, Saver saver) {
            string counterFile = GetCounterFileName(run);
            string runCheckpointDir = Path.Combine(Gpt2Checkpoints.CheckpointDir, run);
            Directory.CreateDirectory(runCheckpointDir);
            Debug.WriteLine("Saving " + Path.Combine(runCheckpointDir, Invariant($"model-{counter}")));
            saver.save(session,
                Path.Combine(runCheckpointDir, "model"),
                global_step: counter);
            File.WriteAllText(path: counterFile, contents: Invariant($"{counter}"));
        }
    }
}
