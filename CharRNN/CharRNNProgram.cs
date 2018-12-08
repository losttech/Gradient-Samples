namespace CharRNN {
    using System;
    using System.Diagnostics;
    using System.IO;
    using Newtonsoft.Json;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.summary;
    using tensorflow.train;

    static class CharRNNProgram {
        static void Main() {
            // ported from https://github.com/sherjilozair/char-rnn-tensorflow
        }

        static void Sample(string saveDir, string prime) {
            var savedArgs = JsonConvert.DeserializeObject<CharRNNModelParameters>(File.ReadAllText(Path.Combine(saveDir, "charRNN.config")));
            var (chars, vocabulary) = LoadCharsVocabulary(Path.Combine(saveDir, "chars_vocab"));
            prime = string.IsNullOrEmpty(prime) ? chars[0] : prime;
            var model = new CharRNNModel(savedArgs, training: false);
            new Session().UseSelf(session => {
                tf.global_variables_initializer().run();
                var saver = new Saver(tf.global_variables());
                var checkpoint = tf.train.get_checkpoint_state(saveDir);
                if (checkpoint?.model_checkpoint_path != null) {
                    saver.restore(session, checkpoint.model_checkpoint_path);
                    Console.WriteLine(model.Sample(session, chars, vocabulary, prime: prime));
                }
            });
        }

        static void Train(string dataDir, int batchSize, int seqLength) {
            var dataLoader = new TextLoader(dataDir, batchSize, seqLength);
            var args = new CharRNNModelParameters(parse from command line);
            args.VocabularySize = dataLoader.vocabularySize;

            var model = new CharRNNModel(args, training: true);
            new Session().UseSelf(session => {
            var summaries = tf.summary.merge_all();
            var writer = new FileWriter(Path.Combine(logDir, DateTime.Now.ToString("s")));
            writer.add_graph(session.graph);

            session.run(new dynamic[] { tf.global_variables_initializer() });
            var saver = new Saver(tf.global_variables());
            if (!string.IsNullOrEmpty(initFrom))
                saver.restore(session, checkpoint);
            for (int e = 0; e < epochs; e++) {
                session.run(tf.assign(model.lr, learningRate * (Math.Pow(decayRate, e))));
                dataLoader.ResetBatchPointer();
                var state = session.run(model.initialState);
                var stopwatch = Stopwatch.StartNew();
                for (int b = 0; b < dataLoader.batchCount; b++) {
                    stopwatch.Restart();
                    var (x, y) = dataLoader.NextBatch();
                    var feed = new PythonDict<dynamic, dynamic{
                        [model.inputData] = x,
                        [model.targets] = y,
                    };
                    foreach (var (i, (c, h)) in model.initialState) {
                        feed[c] = state[i].c;
                        feed[h] = state[i].h;
                    }

                    var step = session.run(new dynamic[] { summaries, model.cost, model.finalState, model.trainOp }, feed);
                    writer.add_summary(step[0], e * dataLoader.batchCount + b);

                    var time = stopwatch.Elapsed;
                    Console.WriteLine($"{e * dataLoader.batchCount + b}/{epochs * dataLoader.batchCount} (epoch {e}), " +
                        $"train loss = {step[1]}, time/batch = {time}");
                    if (((e * dataLoader.batchCount + b) % saveEvery == 0) ||
                        (e == epochs - 1 && b == dataLoader.batchCount - 1)) {
                            string checkpointPath = Path.Combine(saveDir, "model.ckpt");
                            saver.save(session, checkpointPath, global_step: e * dataLoader.batchCount + b);
                            Console.WriteLine("model saved to " + checkpointPath);
                        }
                    }
                }
            });
        }
    }
}
