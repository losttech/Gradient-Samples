namespace LostTech.Gradient.Samples {
    using System;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Runtime.InteropServices;

    static class BitmapTools {
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

        // default .NET upscaling tries to interpolate, which we avoid here
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

        public static void SetGreyscalePalette(Bitmap bitmap) {
            ColorPalette pal = bitmap.Palette;

            for (int i = 0; i < 256; i++) {
                pal.Entries[i] = Color.FromArgb(255, i, i, i);
            }

            bitmap.Palette = pal;
        }
    }
}
