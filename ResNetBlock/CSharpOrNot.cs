namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Text.RegularExpressions;
    using numpy;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.core.protobuf.config_pb2;
    using tensorflow.keras;
    using tensorflow.keras.callbacks;
    using tensorflow.keras.layers;
    using tensorflow.keras.optimizers;
    using static System.Linq.Enumerable;

    public static class CSharpOrNot
    {
        public static Model CreateModel(int classCount) {
            var activation = tf.keras.activations.elu_fn;
            const int filterCount = 8;
            int[] resNetFilters = { filterCount, filterCount, filterCount };
            var model = new Sequential(new Layer[] {
                new Dropout(rate: 0.05),
                Conv2D.NewDyn(filters: filterCount, kernel_size: 5, padding: "same"),
                Activation.NewDyn(activation),
                new MaxPool2D(pool_size: 2),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new MaxPool2D(),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new MaxPool2D(),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new AvgPool2D(pool_size: 2),
                new Flatten(),
                new Dense(units: classCount, activation: tf.nn.softmax_fn),
            });

            model.compile(
                optimizer: new Adam(),
                loss: tf.keras.losses.sparse_categorical_crossentropy_fn,
                metrics: new dynamic[] { "accuracy" });

            return model;
        }

        const int Epochs = 600;
        const int BatchSize = 32;
        public const int Width = 64, Height = 64;
        public static readonly Size Size = new Size(Width, Height);
        // being opinionated here
        const string Tab = "    ";
        const char Whitespace = '\u00FF';
        const float TestSplit = 0.1f;
        const int TrainingSamples = 400000, TestSamples = 20000;
        const int SamplePart = 1;

        public static void Run(params string[] directories) {
            var filesByExtension = ReadCodeFiles(directories,
                includeExtensions: IncludeExtensions,
                codeFilter: lines => lines.Length >= 10);
            Train(filesByExtension);
        }
        public static void Train(List<string[]>[] filesByExtension) {
            GradientSetup.EnsureInitialized();

            dynamic config = config_pb2.ConfigProto();
            config.gpu_options.allow_growth = true;
            tf.keras.backend.set_session(Session.NewDyn(config: config));

            var random = new Random(4242);
            tf.set_random_seed(4242);

            var test = Split(random, filesByExtension, TestSplit, out filesByExtension);

            byte[] trainData = Sample(random, filesByExtension, Size, TrainingSamples/SamplePart,
                out int[] trainValues);
            byte[] testData = Sample(random, test, Size, TestSamples / SamplePart,
                out int[] testValues);

            //var checkpoint = new ModelCheckpoint(
            //    filepath: Path.Combine(Environment.CurrentDirectory, "weights.{epoch:02d}-{val_loss:.4f}.hdf5"),
            //    save_best_only: true, save_weights_only: true);
            var checkpointBest = new ModelCheckpoint(
                filepath: Path.Combine(Environment.CurrentDirectory, "weights.best.hdf5"),
                save_best_only: true, save_weights_only: true);
            // not present in 1.10, but present in 1.14
            ((dynamic)checkpointBest).save_freq = "epoch";

            Console.Write("loading data to TensorFlow...");
            var timer = Stopwatch.StartNew();
            ndarray<float> @in = InputToNumPy(trainData, sampleCount: TrainingSamples / SamplePart,
                width: Width, height: Height);
            ndarray<int> expectedOut = OutputToNumPy(trainValues);
            Console.WriteLine($"OK in {timer.ElapsedMilliseconds / 1000}s");

            string runID = DateTime.Now.ToString("s").Replace(':', '-');
            string logDir = Path.Combine(".", "logs", runID);
            Directory.CreateDirectory(logDir);
            var tensorboard = new TensorBoard(log_dir: logDir);

            var model = CreateModel(classCount: IncludeExtensions.Length);
            model.build(new TensorShape(null, Height, Width, 1));
            model.summary();

            GC.Collect();

            var valIn = InputToNumPy(testData, sampleCount: TestSamples / SamplePart, width: Width, height: Height);
            var valOut = OutputToNumPy(testValues);

            fit(model, @in, expectedOut,
                batchSize: BatchSize,
                epochs: Epochs,
                callbacks: new ICallback[] {
                    checkpointBest,
                    tensorboard,
                }, validationInput: valIn, validationTarget: valOut);

            model.save_weights(Path.GetFullPath("weights.final.hdf5"));

            var fromCheckpoint = CreateModel(classCount: IncludeExtensions.Length);
            fromCheckpoint.build(new TensorShape(null, Height, Width, 1));
            fromCheckpoint.load_weights(Path.GetFullPath("weights.best.hdf5"));

            var evaluationResults = fromCheckpoint.evaluate(@in, expectedOut);
            Console.WriteLine($"reloaded: loss: {evaluationResults[0]} acc: {evaluationResults[1]}");

            evaluationResults = model.evaluate(@in, expectedOut);
            Console.WriteLine($"original: loss: {evaluationResults[0]} acc: {evaluationResults[1]}");
        }

        public static ndarray<float> InputToNumPy(byte[] inputs, int sampleCount, int width, int height)
            => (dynamic)inputs.Select(b => (float)b).ToArray().NumPyCopy()
                .reshape(new[] { sampleCount, height, width, 1 }) / 255.0f;
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

        static List<string[]>[] Split(Random random, List<string[]>[] filesByExtension, float ratio,
            out List<string[]>[] rest)
        {
            var result = new List<string[]>[filesByExtension.Length];
            rest = new List<string[]>[filesByExtension.Length];
            for (int extensionIndex = 0; extensionIndex < filesByExtension.Length; extensionIndex++)
            {
                result[extensionIndex] = Split(random, filesByExtension[extensionIndex], ratio,
                    out rest[extensionIndex]);
            }
            return result;
        }

        static List<T> Split<T>(Random random, ICollection<T> collection, float ratio, out List<T> rest) {
            if (ratio >= 1 || ratio <= 0) throw new ArgumentOutOfRangeException(nameof(ratio));
            int resultLength = (int)(collection.Count * ratio);
            if (resultLength == 0 || resultLength == collection.Count)
                throw new ArgumentException();

            var array = collection.ToArray();
            random.Shuffle(array);

            rest = array.Skip(resultLength).ToList();
            return array.Take(resultLength).ToList();
        }

        static void Shuffle<T>(this Random rng, T[] array) {
            int n = array.Length;
            while (n > 1) {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }

        static void Sample(Random random, List<string[]>[] filesByExtension, Size blockSize,
            byte[] target, out int extensionIndex)
        {
            extensionIndex = random.Next(filesByExtension.Length);
            var files = filesByExtension[extensionIndex];
            int fileIndex = random.Next(files.Count);
            string[] lines = files[fileIndex];
            int y = random.Next(Math.Max(1, lines.Length - 1));
            int x = random.Next(Math.Max(1, lines[y].Length));
            Render(lines, new Point(x, y), blockSize, destination: target);
        }

        static void fit(Model @this, numpy.I_ArrayLike input, numpy.I_ArrayLike targetValues,
            int? stepsPerEpoch = null, int? validationSteps = null,
            int batchSize = 1,
            int epochs = 1,
            TrainingVerbosity verbosity = TrainingVerbosity.ProgressBar,
            IEnumerable<ICallback> callbacks = null,
            numpy.I_ArrayLike validationInput = null,
            numpy.I_ArrayLike validationTarget = null) {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (targetValues == null) throw new ArgumentNullException(nameof(targetValues));
            if (stepsPerEpoch <= 0) throw new ArgumentOutOfRangeException(nameof(stepsPerEpoch));
            if (validationSteps <= 0)
                throw new ArgumentOutOfRangeException(nameof(validationSteps));
            if (validationSteps != null && stepsPerEpoch == null)
                throw new ArgumentException(
                    $"Can't set {nameof(validationSteps)} without setting {nameof(stepsPerEpoch)}",
                    paramName: nameof(validationSteps));

            var validation = validationInput == null && validationTarget == null
                ? (((numpy.I_ArrayLike, numpy.I_ArrayLike)?)null)
                : validationInput != null && validationTarget != null
                    ? (validationInput, validationTarget)
                    : throw new ArgumentException(
                        $"Both (or none) {nameof(validationInput)} and {nameof(validationTarget)} must be provided");

            @this.fit_dyn(input, targetValues,
                epochs: epochs,
                batch_size: batchSize,
                verbose: (int)verbosity,
                callbacks: callbacks,
                validation_data: validation,
                shuffle: false,
                steps_per_epoch: stepsPerEpoch,
                validation_steps: validationSteps
            );
        }

        public static List<string[]>[] ReadCodeFiles(IEnumerable<string> directories,
            string[] includeExtensions,
            Func<string[], bool> codeFilter) {
            var files = Range(0, includeExtensions.Length)
                .Select(_ => new List<string[]>())
                .ToArray();

            foreach (string directory in directories) {
                ReadCodeFiles(files, directory, includeExtensions, codeFilter);
            }

            return files;
        }

        static List<string[]>[] ReadCodeFiles(string directory, string[] includeExtensions,
            Func<string[], bool> codeFilter) {
            var files = Range(0, includeExtensions.Length)
                .Select(_ => new List<string[]>())
                .ToArray();

            ReadCodeFiles(files, directory, includeExtensions, codeFilter);

            return files;
        }

        static void ReadCodeFiles(List<string[]>[] files, string directory, string[] includeExtensions, Func<string[], bool> codeFilter) {
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

        public static string[] ReadCode(string path)
            => File.ReadAllLines(path)
                .Select(line => line.Replace("\t", Tab))
                .Select(line => {
                    var result = new StringBuilder(line.Length);
                    // replace non-ASCII characters with underscore
                    // make all whitespace stand out
                    foreach (char c in line) {
                        result.Append(
                            c <= 32 ? Whitespace
                            : c >= 255 ? '_'
                            : c);
                    }
                    return result.ToString();
                })
                .ToArray();

        public static void Render(string[] lines, Point startingPoint, Size size, byte[] destination) {
            if (size.IsEmpty) throw new ArgumentException();
            if (destination.Length < size.Width * size.Height) throw new ArgumentException();
            if (startingPoint.Y == lines.Length) {
                Array.Fill(destination, (byte)Whitespace);
                return;
            }

            for (int y = 0; y < size.Height; y++) {
                int sourceY = y + startingPoint.Y;
                int destOffset = y * size.Width;
                if (sourceY >= lines.Length) {
                    Array.Fill(destination, (byte)255,
                        startIndex: destOffset,
                        count: size.Width*size.Height - destOffset);
                    return;
                }

                for (int x = 0; x < size.Width; x++) {
                    int sourceX = x + startingPoint.X;
                    if (sourceX >= lines[sourceY].Length) {
                        Array.Fill(destination, (byte)255,
                            startIndex: destOffset,
                            count: size.Width - x);
                        break;
                    }

                    destination[destOffset] = (byte)lines[sourceY][sourceX];
                    destOffset++;
                }
            }
        }

        public static void ToBitmap(byte[] brightness, Bitmap target) {
            if (target.PixelFormat != PixelFormat.Format8bppIndexed)
                throw new NotSupportedException("The only supported pixel format is " + PixelFormat.Format8bppIndexed);

            var bitmapData = target.LockBits(new Rectangle(new Point(), target.Size),
                ImageLockMode.WriteOnly,
                PixelFormat.Format8bppIndexed);

            try {
                Marshal.Copy(source: brightness,
                    startIndex: 0, length: bitmapData.Width * bitmapData.Height,
                    destination: bitmapData.Scan0);
            } finally {
                target.UnlockBits(bitmapData);
            }
        }

        public static void Upscale(Bitmap source, Bitmap target) {
            if (target.Width % source.Width != 0 || target.Height % source.Height != 0)
                throw new ArgumentException();

            int scaleY = target.Height / source.Height;
            int scaleX = target.Width / source.Width;

            var sourceData = source.LockBits(new Rectangle(new Point(), source.Size),
                ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
            try {
                var targetData = target.LockBits(new Rectangle(new Point(), target.Size),
                    ImageLockMode.WriteOnly,
                    PixelFormat.Format8bppIndexed);

                try {
                    for (int sourceY = 0; sourceY < sourceData.Height; sourceY++)
                    for (int sourceX = 0; sourceX < sourceData.Width; sourceX++) {
                        byte brightness = Marshal.ReadByte(sourceData.Scan0,
                            sourceY * sourceData.Width + sourceX);
                        for (int targetY = sourceY * scaleY;
                            targetY < (sourceY + 1) * scaleY;
                            targetY++)
                        for (int targetX = sourceX * scaleX;
                            targetX < (sourceX + 1) * scaleX;
                            targetX++)
                            Marshal.WriteByte(targetData.Scan0,
                                targetY * targetData.Width + targetX,
                                brightness);
                    }
                } finally {
                    target.UnlockBits(targetData);
                }
            } finally {
                source.UnlockBits(sourceData);
            }
        }

        public static void SetGrayscalePalette(Bitmap bitmap) {
            ColorPalette pal = bitmap.Palette;

            for (int i = 0; i < 256; i++) {
                pal.Entries[i] = Color.FromArgb(255, i, i, i);
            }

            bitmap.Palette = pal;
        }

        public static readonly string[] IncludeExtensions = {
            ".cs",
            ".py",
            ".h",
            ".cc",
            ".c",
            ".tcl",
            ".java",
            ".sh",
        };
    }
}
