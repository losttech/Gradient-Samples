namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Text.RegularExpressions;
    using numpy;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.callbacks;
    using tensorflow.keras.layers;
    using tensorflow.keras.optimizers;
    using static System.Linq.Enumerable;

    public static class CSharpOrNot
    {
        public static Model CreateModel(int classCount) {
            var activation = tf.keras.activations.elu_fn;
            const int filterCount = 32;
            int[] resNetFilters = {filterCount, filterCount, filterCount};
            var model = new Sequential(new Layer[] {
                Conv2D.NewDyn(filters: filterCount, kernel_size: 5, padding: "same"),
                Activation.NewDyn(activation),
                new MaxPool2D(pool_size: 2),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new ResNetBlock(kernelSize: 3, filters: resNetFilters, activation: activation),
                new AvgPool2D(pool_size: 2),
                new Flatten(),
                new Dense(units: 32, activation: activation),
                new Dense(units: classCount, activation: activation),
            });

            model.compile(
                optimizer: new Adam(epsilon: 1e-5),
                loss: tf.keras.losses.mean_squared_error_fn,
                metrics: new dynamic[] { "accuracy" });

            return model;
        }

        const int Epochs = 100;
        const int BatchSize = 1000;
        public const int Width = 64, Height = 64;
        public static readonly Size Size = new Size(Width, Height);
        // being opinionated here
        const string Tab = "    ";
        const char Whitespace = '\u00FF';
        const float ValidationSplit = 0.1f, TestSplit = 0.1f;
        const int TrainingSamples = 200000, TestSamples = 10000, ValidationSamples = 10000;
        const int SamplePart = 10;

        public static void Run(string directory) {
            GradientSetup.EnsureInitialized();

            var filesByExtension = ReadCodeFiles(directory,
                includeExtensions: IncludeExtensions,
                codeFilter: lines => lines.Length >= 10);

            var random = new Random(42);
            tf.set_random_seed(42);

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

            Console.Write("loading data to TensorFlow...");
            var timer = Stopwatch.StartNew();
            ndarray<float> @in = InputToNumPy(trainData, sampleCount: TrainingSamples / SamplePart,
                width: Width, height: Height);
            ndarray<float> expectedOut = OutputToNumPy(trainValues);
            Console.WriteLine($"OK in {timer.ElapsedMilliseconds / 1000}s");

            var model = CreateModel(classCount: IncludeExtensions.Length);
            model.build(new TensorShape(null, Height, Width, 1));
            model.summary();
            model.fit_dyn(@in, expectedOut, epochs: Epochs, shuffle: false, batch_size: BatchSize,
                validation_split: 0.1, callbacks: new ICallback[] {
                    //checkpointBest
                },
                verbose: TrainingVerbosity.LinePerEpoch);

            var fromCheckpoint = CreateModel(classCount: IncludeExtensions.Length);
            fromCheckpoint.build(new TensorShape(null, Height, Width, 1));
            fromCheckpoint.load_weights(Path.GetFullPath("weights.best.hdf5"));

            @in = InputToNumPy(testData, sampleCount: TestSamples/SamplePart, width: Width, height: Height);
            expectedOut = OutputToNumPy(testValues);

            var evaluationResults = fromCheckpoint.evaluate(@in, expectedOut);
            Console.WriteLine($"reloaded: loss: {evaluationResults[0]} acc: {evaluationResults[1]}");

            evaluationResults = model.evaluate(@in, expectedOut);
            Console.WriteLine($"original: loss: {evaluationResults[0]} acc: {evaluationResults[1]}");
        }

        public static ndarray<float> InputToNumPy(byte[] inputs, int sampleCount, int width, int height)
            => (dynamic)inputs.Select(b => (float)b).ToArray().NumPyCopy()
                .reshape(new[] { sampleCount, height, width, 1 }) / 255.0f;
        static ndarray<float> OutputToNumPy(int[] expectedOutputs)
            => (ndarray<float>)np.eye(IncludeExtensions.Length, dtype: PythonClassContainer<float32>.Instance)
                .__getitem__(expectedOutputs.ToNumPyArray().reshape(-1));

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

        static List<string[]>[] ReadCodeFiles(string directory, string[] includeExtensions,
            Func<string[], bool> codeFilter) {
            var files = Range(0, includeExtensions.Length)
                .Select(_ => new List<string[]>())
                .ToArray();

            foreach (string filePath in Directory.EnumerateFiles(directory, "*.*", SearchOption.AllDirectories)) {
                string extension = Path.GetExtension(filePath);
                int extensionIndex = Array.IndexOf(includeExtensions, extension);
                if (extensionIndex < 0) continue;
                string[] code = ReadCode(filePath);
                if (!codeFilter(code)) continue;
                files[extensionIndex].Add(code);
            }

            return files;
        }

        public static string[] ReadCode(string path)
            => File.ReadAllLines(path)
                .Select(line => line.Replace("\t", Tab))
                // replace non-ASCII characters with underscore
                .Select(line => Regex.Replace(line, @"[^\u0000-\u007F]", "_"))
                // make all whitespace stand out
                .Select(line => Regex.Replace(line, @"[\u0000-\u0020]", Whitespace.ToString()))
                .ToArray();

        public static void Render(string[] lines, Point startingPoint, Size size, byte[] destination) {
            if (size.IsEmpty) throw new ArgumentException();
            if (destination.Length < size.Width * size.Height) throw new ArgumentException();
            if (startingPoint.Y >= lines.Length) throw new ArgumentException();

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
            ".dart",
            ".go",
            ".java",
            ".hs", // Haskell
            ".m", // Objective-C
            ".py",
            ".rs", // Rust
            ".u", // Unison
            ".xml",
        };
    }
}
