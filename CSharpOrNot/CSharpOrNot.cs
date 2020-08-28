namespace LostTech.Gradient.Samples {
    using System;
    using System.Drawing;
    using System.IO;
    using System.Linq;
    using System.Text;
    using numpy;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;

    static class CSharpOrNot
    {
        public static Model CreateModel(int classCount) {
            var activation = tf.keras.activations.elu_fn;
            const int filterCount = 8;
            int[] resNetFilters = { filterCount, filterCount, filterCount };
            return new Sequential(new Layer[] {
                new Dropout(rate: 0.05),
                Conv2D.NewDyn(filters: filterCount, kernel_size: 5, padding: "same"),
                new Activation(activation),
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
        }

        public const int Width = 64, Height = 64;
        public static readonly Size Size = new Size(Width, Height);
        // being opinionated here
        const string Tab = "    ";
        const char Whitespace = '\u00FF';
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

        public static ndarray<float> GreyscaleImageBytesToNumPy(byte[] inputs, int imageCount, int width, int height)
            => (ndarray<float>)inputs.Select(b => b/255.0f).ToArray().ToNumPyArray()
                .reshape(new[] { imageCount, height, width, 1 });

        public static string[] ReadCode(string filePath)
            => File.ReadAllLines(filePath)
                .Select(line => line.Replace("\t", Tab))
                .Select(line => {
                    var result = new StringBuilder(line.Length);
                    // replace non-ASCII characters with underscore
                    // also make all whitespace stand out
                    foreach (char c in line) {
                        result.Append(
                            c <= 32 ? Whitespace
                            : c >= 255 ? '_'
                            : c);
                    }
                    return result.ToString();
                })
                .ToArray();

        /// <summary>
        /// Copies a rectangular block of text into a byte array
        /// </summary>
        public static void RenderTextBlockToGreyscaleBytes(string[] lines,
            Point startingPoint, Size size,
            byte[] destination)
        {
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
    }
}
