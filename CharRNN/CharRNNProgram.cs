namespace CharRNN {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using CommandLine;
    using Newtonsoft.Json;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.summary;
    using tensorflow.train;

    static class CharRNNProgram {
        const string ConfigFileName = "charRNN.config";
        const string CharsVocabularyFileName = "chars_vocab";

        static int Main(string[] args) {
            throw new NotImplementedException("Work in progress");
            // ported from https://github.com/sherjilozair/char-rnn-tensorflow
            return Parser.Default.ParseArguments<CharRNNTrainingParameters>(args)
                .MapResult(Train, errors => 1);
        }

        static void Sample(string saveDir, string prime) {
            var savedArgs = JsonConvert.DeserializeObject<CharRNNModelParameters>(File.ReadAllText(Path.Combine(saveDir, ConfigFileName)));
            var (chars, vocabulary) = LoadCharsVocabulary(Path.Combine(saveDir, CharsVocabularyFileName));
            prime = string.IsNullOrEmpty(prime) ? chars[0].ToString() : prime;
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

        static int Train(CharRNNTrainingParameters args) {
            var dataLoader = new TextLoader(args.dataDir, args.BatchSize, args.SeqLength);
            args.VocabularySize = dataLoader.vocabularySize;
            dynamic checkpoint = null;
            if (!string.IsNullOrEmpty(args.initFrom)) {
                checkpoint = tf.train.latest_checkpoint(args.initFrom);
                var savedArgs =
                    JsonConvert.DeserializeObject<CharRNNModelParameters>(
                        File.ReadAllText(Path.Combine(args.initFrom, ConfigFileName)));
                Trace.Assert(savedArgs.ModelType == args.ModelType);
                Trace.Assert(savedArgs.RNNSize == args.RNNSize);
                Trace.Assert(savedArgs.LayerCount == args.LayerCount);
                Trace.Assert(savedArgs.SeqLength == args.SeqLength);

                var (chars, vocabulary) = LoadCharsVocabulary(Path.Combine(args.saveDir, CharsVocabularyFileName));
                Trace.Assert(dataLoader.chars.SequenceEqual(chars));
                Trace.Assert(dataLoader.vocabulary.SequenceEqual(vocabulary));
            }

            Directory.CreateDirectory(args.saveDir);
            File.WriteAllText(Path.Combine(args.saveDir, ConfigFileName), JsonConvert.SerializeObject(args));
            File.WriteAllText(Path.Combine(args.saveDir, CharsVocabularyFileName), JsonConvert.SerializeObject((dataLoader.chars, dataLoader.vocabulary)));

            var model = new CharRNNModel(args, training: true);

            new Session().UseSelf(session => {
                var summaries = tf.summary.merge_all();
                var writer = new FileWriter(Path.Combine(args.logDir, DateTime.Now.ToString("s")));
                writer.add_graph(session.graph);

                session.run(new dynamic[] { tf.global_variables_initializer() });
                var saver = new Saver(tf.global_variables());
                if (checkpoint != null)
                    saver.restore(session, checkpoint);

                for (int e = 0; e < args.epochs; e++) {
                    session.run(tf.assign(model.lr, new dynamic[] { args.learningRate * Math.Pow(args.decayRate, e) }));
                    dataLoader.ResetBatchPointer();
                    var state = session.run(model.initialState.Cast<object>());
                    var stopwatch = Stopwatch.StartNew();
                    for (int b = 0; b < dataLoader.batchCount; b++) {
                        stopwatch.Restart();
                        var (x, y) = dataLoader.NextBatch();
                        var feed = new PythonDict<dynamic, dynamic> {
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
                        Console.WriteLine($"{e * dataLoader.batchCount + b}/{args.epochs * dataLoader.batchCount} (epoch {e}), " +
                            $"train loss = {step[1]}, time/batch = {time}");
                        if (((e * dataLoader.batchCount + b) % args.saveEvery == 0)
                            || (e == args.epochs - 1 && b == dataLoader.batchCount - 1)) {
                            string checkpointPath = Path.Combine(args.saveDir, "model.ckpt");
                            saver.save(session, checkpointPath, global_step: e * dataLoader.batchCount + b);
                            Console.WriteLine("model saved to " + checkpointPath);
                        }
                    }
                }
            });
            return 1;
        }

        static (List<char>, Dictionary<char, int>) LoadCharsVocabulary(string path)
            => JsonConvert.DeserializeObject<(List<char>, Dictionary<char, int>)>(File.ReadAllText(path));
    }
}
