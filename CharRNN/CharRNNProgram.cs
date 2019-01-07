namespace CharRNN {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using CommandLine;
    using Gradient;
    using Newtonsoft.Json;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.summary;
    using tensorflow.train;

    static class CharRNNProgram {
        const string ConfigFileName = "charRNN.config";
        const string CharsVocabularyFileName = "chars_vocab";

        static int Main(string[] args) {
            // ported from https://github.com/sherjilozair/char-rnn-tensorflow
            return Parser.Default.ParseArguments<CharRNNTrainingParameters, CharRNNSamplingParameters>(args)
                .MapResult(
                (CharRNNTrainingParameters train) => Train(train),
                (CharRNNSamplingParameters sample) => Sample(sample),
                _ => 1);
        }

        static int Sample(CharRNNSamplingParameters args) {
            var savedArgs = JsonConvert.DeserializeObject<CharRNNModelParameters>(File.ReadAllText(Path.Combine(args.saveDir, ConfigFileName)));
            var (chars, vocabulary) = LoadCharsVocabulary(Path.Combine(args.saveDir, CharsVocabularyFileName));
            string prime = string.IsNullOrEmpty(args.prime) ? chars[0].ToString() : args.prime;
            var model = new CharRNNModel(savedArgs, training: false);
            new Session().UseSelf(session => {
                tf.global_variables_initializer().run();
                var saver = new Saver(tf.global_variables());
                var checkpoint = tf.train.get_checkpoint_state(args.saveDir);
                if (checkpoint?.model_checkpoint_path != null) {
                    saver.restore(session, checkpoint.model_checkpoint_path);
                    Console.WriteLine(model.Sample(session, chars, vocabulary, prime: prime, num: args.count));
                }
            });
            return 0;
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
                var writer = new FileWriter(Path.Combine(args.logDir, DateTime.Now.ToString("s").Replace(':', '-')));
                writer.add_graph(session.graph);

                session.run(new dynamic[] { tensorflow.tf.global_variables_initializer() });
                var globals = tf.global_variables();
                var saver = new Saver(globals);
                if (checkpoint != null)
                    saver.restore(session, checkpoint);

                int totalNumberOfBatches = args.epochs * dataLoader.batchCount;

                for (int epoch = 0; epoch < args.epochs; epoch++) {
                    session.run(new[] { tf.assign(
                        model.learningRate,
                        tf.constant(args.learningRate * Math.Pow(args.decayRate, epoch)))
                    });
                    dataLoader.ResetBatchPointer();
                    var state = session.run(model.initialState.Items().Cast<object>());
                    var stopwatch = Stopwatch.StartNew();
                    for (int batch = 0; batch < dataLoader.batchCount; batch++) {
                        stopwatch.Restart();
                        var (input, targets) = dataLoader.NextBatch();
                        var feed = new PythonDict<dynamic, dynamic> {
                            [model.inputData] = input,
                            [model.targets] = targets,
                        };
                        foreach (var (i, tuple) in model.initialState.Items().Enumerate()) {
                            feed[tuple.c] = state[i].c;
                            feed[tuple.h] = state[i].h;
                        }

                        var step = session.run(new dynamic[] { summaries, model.cost, model.finalState, model.trainOp }, feed);

                        int currentBatch = (epoch * dataLoader.batchCount) + batch;
                        writer.add_summary(step[0], currentBatch);

                        var time = stopwatch.Elapsed;
                        Console.WriteLine(
                            $"{currentBatch}/{totalNumberOfBatches} " +
                            $"(epoch {epoch}), " +
                            $"train loss = {step[1]}, time/batch = {time}");
                        if ((currentBatch % args.saveEvery == 0)
                            || (epoch == args.epochs - 1 && batch == dataLoader.batchCount - 1)) {
                            string checkpointPath = Path.Combine(args.saveDir, "model.ckpt");
                            saver.save(session, checkpointPath, global_step: currentBatch);
                            Console.WriteLine("model saved to " + checkpointPath);
                        }
                    }
                }
            });
            return 0;
        }

        static (List<char>, Dictionary<char, int>) LoadCharsVocabulary(string path)
            => JsonConvert.DeserializeObject<(List<char>, Dictionary<char, int>)>(File.ReadAllText(path));
    }
}
