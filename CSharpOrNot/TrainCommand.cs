namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.IO;
    using System.Linq;
    using JetBrains.Annotations;
    using ManyConsole.CommandLineUtils;
    using numpy;
    using tensorflow;
    using tensorflow.keras.callbacks;
    using tensorflow.keras.optimizers;
    using static CSharpOrNot;

    [UsedImplicitly]
    class TrainCommand: ConsoleCommand {
        // Set this to 10 or 100 to do a quick run
        const int SamplePart = 1;

        public override int Run(string[] remainingArguments) {
            string[] directories = remainingArguments;
            if (directories is null || directories.Length == 0) {
                Console.Error.WriteLine("Must specify at least one directory!");
                return -1;
            }
            var filesByExtension = ReadCodeFiles(directories,
                includeExtensions: IncludeExtensions,
                codeFilter: lines => lines.Length >= 10);
            if (filesByExtension.Sum(fileList => fileList.Count) == 0) {
                Console.Error.WriteLine("Found no files for training!");
                return -2;
            }
            return this.Train(filesByExtension);
        }

        int Train(List<string[]>[] filesByExtension) {
            var random = this.Seed is null ? new Random() : new Random(this.Seed.Value);
            if (this.Seed != null)
                tf.set_random_seed(this.Seed.Value);

            var test = Split(random, filesByExtension, this.TestSplit, out filesByExtension);

            byte[] trainData = Sample(random, filesByExtension, CSharpOrNot.Size,
                count: this.TrainingSamples / SamplePart, out int[] trainValues);
            byte[] testData = Sample(random, test, CSharpOrNot.Size,
                count: this.TestSamples / SamplePart, out int[] testValues);

            //var checkpoint = new ModelCheckpoint(
            //    filepath: Path.Combine(Environment.CurrentDirectory, "weights.{epoch:02d}-{val_loss:.4f}.hdf5"),
            //    save_best_only: true, save_weights_only: true);
            var checkpointBest = new ModelCheckpoint(
                filepath: Path.Combine(Environment.CurrentDirectory, "weights.best.hdf5"),
                save_best_only: true, save_weights_only: true);
            // not present in 1.10, but present in 1.14
            ((dynamic)checkpointBest).save_freq = "epoch";

            ndarray<float> @in = GreyscaleImageBytesToNumPy(trainData, imageCount: this.TrainingSamples / SamplePart,
                width: Width, height: Height);
            ndarray<int> expectedOut = OutputToNumPy(trainValues);

            string runID = DateTime.Now.ToString("s").Replace(':', '-');
            string logDir = Path.Combine(".", "logs", runID);
            Directory.CreateDirectory(logDir);
            var tensorboard = new TensorBoard(log_dir: logDir);

            var model = CreateModel(classCount: IncludeExtensions.Length);
            model.compile(
                optimizer: new Adam(),
                loss: tf.keras.losses.sparse_categorical_crossentropy_fn,
                metrics: new dynamic[] { "accuracy" });
            model.build(input_shape: new TensorShape(null, Height, Width, 1));
            model.summary();

            GC.Collect();

            var validationInputs = GreyscaleImageBytesToNumPy(testData,
                imageCount: this.TestSamples / SamplePart,
                width: Width, height: Height);
            var validationOutputs = OutputToNumPy(testValues);

            model.fit(@in, expectedOut,
                batchSize: this.BatchSize, epochs: this.Epochs,
                callbacks: new ICallback[] {
                    checkpointBest,
                    tensorboard,
                }, validationInput: validationInputs, validationTarget: validationOutputs);

            model.save_weights(Path.GetFullPath("weights.final.hdf5"));

            var fromCheckpoint = CreateModel(classCount: IncludeExtensions.Length);
            fromCheckpoint.build(new TensorShape(null, Height, Width, 1));
            fromCheckpoint.load_weights(Path.GetFullPath("weights.best.hdf5"));

            var evaluationResults = fromCheckpoint.evaluate(@in, expectedOut);
            Console.WriteLine($"reloaded: loss: {evaluationResults[0]} acc: {evaluationResults[1]}");

            evaluationResults = model.evaluate(@in, expectedOut);
            Console.WriteLine($"original: loss: {evaluationResults[0]} acc: {evaluationResults[1]}");

            return 0;
        }

        public int Epochs { get; set; } = 600;
        public int BatchSize { get; set; } = 32;
        public float TestSplit { get; set; } = 0.1f;
        public int TrainingSamples { get; set; } = 400_000;
        public int TestSamples { get; set; } = 20_000;
        public int? Seed { get; set; }

        public TrainCommand() {
            this.IsCommand("train");

            this.HasOption("e|epochs=", "How many epochs to train for",
                (int epochs) => this.Epochs = epochs);
            this.HasOption("b|batch-size=", "Size of a minibatch for training",
                (int batchSize) => this.BatchSize = batchSize);
            this.HasOption("test-split=", "Percentage of the files to use for testing every epoch",
                (float testSplit) => this.TestSplit = testSplit);
            this.HasOption("s|training-samples=", "How many samples to generate for training",
                (int trainingSamples) => this.TrainingSamples = trainingSamples);
            this.HasOption("t|test-samples=", "How many samples to generate for testing",
                (int testSamples) => this.TestSamples = testSamples);
            this.HasOption("seed=", "Attempts to make training reproducible by using a fixed random seed",
                (int seed) => this.Seed = seed);

            this.AllowsAnyAdditionalArguments("directories with training source code");
        }

        static ndarray<int> OutputToNumPy(int[] expectedOutputs)
            => expectedOutputs.ToNumPyArray();

        static byte[] Sample(Random random, List<string[]>[] filesByExtension, Size size, int count,
            out int[] extensionIndices) {
            byte[] result = new byte[count * size.Width * size.Height];
            byte[] sampleBytes = new byte[size.Width * size.Height];
            extensionIndices = new int[count];
            for (int sampleIndex = 0; sampleIndex < count; sampleIndex++) {
                Sample(random, filesByExtension, size, sampleBytes, out extensionIndices[sampleIndex]);
                Array.Copy(sampleBytes, sourceIndex: 0, length: sampleBytes.Length,
                    destinationArray: result, destinationIndex: sampleIndex*sampleBytes.Length);
            }

            return result;
        }

        static void Sample(Random random, List<string[]>[] filesByExtension, Size blockSize,
            byte[] target, out int extensionIndex)
        {
            var files = random.Next(filesByExtension, out extensionIndex);
            string[] lines = random.Next(files);
            int y = random.Next(Math.Max(1, lines.Length - 1));
            int x = random.Next(Math.Max(1, lines[y].Length));
            RenderTextBlockToGreyscaleBytes(lines, new Point(x, y), blockSize, destination: target);
        }

        public static List<string[]>[] Split(Random random, List<string[]>[] filesByExtension,
            float ratio, out List<string[]>[] rest)
        {
            var result = new List<string[]>[filesByExtension.Length];
            rest = new List<string[]>[filesByExtension.Length];

            for (int extensionIndex = 0; extensionIndex < filesByExtension.Length; extensionIndex++) {
                string[][] files = filesByExtension[extensionIndex].ToArray();
                random.Shuffle(files);
                result[extensionIndex] = DataTools.Split(filesByExtension[extensionIndex], ratio,
                    out rest[extensionIndex]);
            }
            return result;
        }

        public static List<string[]>[] ReadCodeFiles(
            IEnumerable<string> directories,
            string[] includeExtensions,
            Func<string[], bool> codeFilter)
        {
            var files = Enumerable.Range(0, includeExtensions.Length)
                .Select(_ => new List<string[]>())
                .ToArray();

            foreach (string directory in directories) {
                ReadCodeFiles(files, directory, includeExtensions, codeFilter);
            }

            return files;
        }

        static void ReadCodeFiles(List<string[]>[] files, string directory,
                                  string[] includeExtensions, Func<string[], bool> codeFilter) {
            foreach (string filePath in Directory.EnumerateFiles(directory, "*.*", SearchOption.AllDirectories)) {
                if (filePath.Contains(
                    Path.DirectorySeparatorChar + "out" + Path.DirectorySeparatorChar))
                    continue;
                string extension = Path.GetExtension(filePath);
                int extensionIndex = Array.IndexOf(includeExtensions, extension);
                if (extensionIndex < 0) continue;
                string[] code = ReadCode(filePath);
                if (!codeFilter(code)) continue;
                files[extensionIndex].Add(code);
            }
        }
    }
}
