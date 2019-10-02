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
    using static System.Linq.Enumerable;

    static class CSharpOrNot
    {
        const int Width = 64, Height = 64;
        const int CharWidth = 10, CharHeight = 16;
        // being opinionated here
        const string Tab = "    ";
        const char Whitespace = '\u00FF';

        public static void Run(string directory) {
            var filesByExtension = ReadCodeFiles(directory,
                includeExtensions: IncludeExtensions,
                codeFilter: lines => lines.Length >= 10);

            var random = new Random();

            var blockSize = new Size(Width, Height);

            byte[] codeBytes = new byte[Width*Height];
            var renderTarget = new Bitmap(Width, Height, PixelFormat.Format8bppIndexed);
            SetGrayscalePalette(renderTarget);
            var output = new Bitmap(Width * CharWidth, Height * CharHeight, PixelFormat.Format8bppIndexed);
            SetGrayscalePalette(output);

            int extensionIndex = random.Next(filesByExtension.Length);
            while (true) {
                var files = filesByExtension[extensionIndex];
                int fileIndex = random.Next(files.Count);
                string[] lines = files[fileIndex];
                int y = random.Next(Math.Max(1, lines.Length - Height));
                int x = random.Next(Math.Max(1, (int)lines.Average(line => line.Length) - Width));
                Render(lines, new Point(x, y), blockSize, destination: codeBytes);
                ToBitmap(codeBytes, target: renderTarget);
                Upscale(renderTarget, output);
                break;
            }

            output.Save("code.png", ImageFormat.Png);
            var startInfo = new ProcessStartInfo(Path.GetFullPath("code.png")) {
                UseShellExecute = true,
                Verb = "open",
            };
            Process.Start(startInfo);
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

        static string[] ReadCode(string path)
            => File.ReadAllLines(path)
                .Select(line => line.Replace("\t", Tab))
                // replace non-ASCII characters with underscore
                .Select(line => Regex.Replace(line, @"[^\u0000-\u007F]", "_"))
                // make all whitespace stand out
                .Select(line => Regex.Replace(line, @"[\u0000-\u0020]", Whitespace.ToString()))
                .ToArray();

        static void Render(string[] lines, Point startingPoint, Size size, byte[] destination) {
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

        static void ToBitmap(byte[] brightness, Bitmap target) {
            if (target.PixelFormat != PixelFormat.Format8bppIndexed)
                throw new NotSupportedException("The only supported pixel format is " + PixelFormat.Format8bppIndexed);

            var bitmapData = target.LockBits(new Rectangle(new Point(), target.Size),
                ImageLockMode.WriteOnly,
                PixelFormat.Format8bppIndexed);

            Marshal.Copy(source: brightness,
                startIndex: 0, length: bitmapData.Width * bitmapData.Height,
                destination: bitmapData.Scan0);

            target.UnlockBits(bitmapData);
        }

        static void Upscale(Bitmap source, Bitmap target) {
            if (target.Width % source.Width != 0 || target.Height % source.Height != 0)
                throw new ArgumentException();

            int scaleY = target.Height / source.Height;
            int scaleX = target.Width / source.Width;

            var sourceData = source.LockBits(new Rectangle(new Point(), source.Size),
                ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
            var targetData = target.LockBits(new Rectangle(new Point(), target.Size),
                ImageLockMode.WriteOnly,
                PixelFormat.Format8bppIndexed);

            for (int sourceY = 0; sourceY < sourceData.Height; sourceY++)
            for (int sourceX = 0; sourceX < sourceData.Width; sourceX++) {
                byte brightness = Marshal.ReadByte(sourceData.Scan0, sourceY * sourceData.Width + sourceX);
                for(int targetY = sourceY*scaleY; targetY < (sourceY+1)*scaleY; targetY++)
                for(int targetX = sourceX*scaleX; targetX < (sourceX+1)*scaleX; targetX++)
                    Marshal.WriteByte(targetData.Scan0, targetY*targetData.Width+targetX, brightness);
            }
        }

        static void SetGrayscalePalette(Bitmap bitmap) {
            ColorPalette pal = bitmap.Palette;

            for (int i = 0; i < 256; i++) {
                pal.Entries[i] = Color.FromArgb(255, i, i, i);
            }

            bitmap.Palette = pal;
        }

        static readonly string[] IncludeExtensions = {
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
