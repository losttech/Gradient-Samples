namespace LostTech.Gradient.Samples {
    using System;
    using System.Drawing.Imaging;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using Avalonia;
    using Avalonia.Controls;
    using Avalonia.Interactivity;
    using Avalonia.Markup.Xaml;
    using Avalonia.Media;
    using LostTech.Gradient;
    using LostTech.Gradient.Exceptions;
    using LostTech.TensorFlow;
    using MoreLinq;
    using numpy;
    using tensorflow;
    using tensorflow.keras;
    using static LostTech.Gradient.Samples.CSharpOrNot;
    using Image = Avalonia.Controls.Image;
    using Point = System.Drawing.Point;
    using Bitmap = System.Drawing.Bitmap;
    using PixelFormat = System.Drawing.Imaging.PixelFormat;

    public class CSharpOrNotWindow : Window
    {
        readonly ContentControl languageBox;
        readonly TextBox codeDisplay;
        readonly TextBlock codeWindow;
        readonly TextBlock language;
        readonly Image codeImage;
        readonly Button openFileButton;
        string[] code;
        readonly Model model;
        bool loaded = false;
        public CSharpOrNotWindow() {
            this.InitializeComponent();
#if DEBUG
            this.AttachDevTools();
#endif
            this.codeDisplay = this.Get<TextBox>("CodeDisplay");
            this.codeDisplay.PropertyChanged += this.CodeDisplayOnPropertyChanged;

            this.codeWindow = this.Get<TextBlock>("CodeWindow");
            this.language = this.Get<TextBlock>("Language");
            this.languageBox = this.Get<ContentControl>("LanguageBox");
            this.codeImage = this.Get<Image>("CodeImage");
            this.openFileButton = this.Get<Button>("OpenFileButton");

            BitmapTools.SetGreyscalePalette(this.renderTarget);
            BitmapTools.SetGreyscalePalette(this.output);

            TensorFlowSetup.Instance.EnsureInitialized();

            this.model = CreateModel(classCount: IncludeExtensions.Length);
            this.model.build(new TensorShape(null, CSharpOrNot.Height, CSharpOrNot.Width, 1));

            this.LoadWeights();
        }

        void CodeDisplayOnPropertyChanged(object sender, AvaloniaPropertyChangedEventArgs e) {
            switch (e.Property.Name) {
            case nameof(TextBox.SelectionStart):
                this.UpdateCodeWindows((int)e.NewValue);
                break;
            }
        }

        static readonly char[] newlines = {'\n','\r'};

        const int CharWidth = 10, CharHeight = 14;

        readonly Bitmap renderTarget = new Bitmap(CSharpOrNot.Width, CSharpOrNot.Height, PixelFormat.Format8bppIndexed);
        readonly Bitmap output = new Bitmap(CSharpOrNot.Width * CharWidth, CSharpOrNot.Height * CharHeight, PixelFormat.Format8bppIndexed);

        private void UpdateCodeWindows(int newStart) {
            var cursorPos = GetCursorPos(this.codeDisplay.Text, newStart);
            byte[] codeBytes = new byte[CSharpOrNot.Width * CSharpOrNot.Height];
            CSharpOrNot.RenderTextBlockToGreyscaleBytes(this.code, cursorPos, CSharpOrNot.Size, codeBytes);
            this.codeWindow.Text = string.Join(Environment.NewLine,
                codeBytes.Select(b => b == 255 ? ' ' : (char)b)
                    .Batch(CSharpOrNot.Width)
                    .Select(line => new string(line.ToArray())));

            BitmapTools.ToBitmap(codeBytes, this.renderTarget);
            BitmapTools.Upscale(this.renderTarget, this.output);

            this.codeImage.Source?.Dispose();
            this.output.Save("code.png", ImageFormat.Png);
            this.codeImage.Source = new Avalonia.Media.Imaging.Bitmap("code.png");

            ndarray @in = GreyscaleImageBytesToNumPy(codeBytes, imageCount: 1,
                width: CSharpOrNot.Width, height: CSharpOrNot.Height);
            var prediction = this.model.predict(@in);
            int extensionIndex = (int)prediction.argmax();
            string extension = IncludeExtensions[extensionIndex].Substring(1);
            bool csharp = extension == "cs";
            this.language.Text = csharp ? "C#" : $"Not C#! ({extension}?)";
            this.languageBox.Background = csharp ? Brushes.Green : Brushes.Red;
        }

        static Point GetCursorPos(string text, int position) {
            int lineStart = -1;
            int y = 0;
            while (true) {
                int newLineStart = text.IndexOfAny(newlines, lineStart + 1);
                if (newLineStart < 0)
                    break;
                if (newLineStart + 1 != text.Length
                    && text[newLineStart] != text[newLineStart+1]
                    && newlines.Contains(text[newLineStart+1])) {
                    newLineStart++;
                }

                if (newLineStart >= position) {
                    break;
                }
                y++;
                lineStart = newLineStart;
            }
            return new Point(x: position - lineStart - 1, y: y);
        }

        private void InitializeComponent() {
            AvaloniaXamlLoader.Load(this);
        }

        async Task LoadWeights() {
            while (!this.loaded) {
                string modelFile = Environment.GetEnvironmentVariable("CS_OR_NOT_WEIGHTS")
                                   ?? (await new OpenFileDialog {
                                       Title = "Load model weights",
                                       InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                                       AllowMultiple = false,
                                   }.ShowAsync(this)).Single();

                if (Path.GetExtension(modelFile) == ".index")
                    modelFile = Path.Combine(
                        Path.GetDirectoryName(modelFile),
                        Path.GetFileNameWithoutExtension(modelFile));

                try {
                    this.model.load_weights(modelFile);
                } catch(ValueError e) {
                    this.Title = this.codeDisplay.Text = e.Message;
                    continue;
                }
                this.model.trainable = false;
                this.loaded = true;
                this.Title = modelFile;
            }

            this.openFileButton.IsEnabled = true;
        }

        async void OpenFileClick(object sender, RoutedEventArgs e) {
            var dialog = new OpenFileDialog {
                Title = "Select code file",
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                AllowMultiple = false,
            };
            string[] files = await dialog.ShowAsync(this);

            if (files.Length == 0) return;
            this.code = ReadCode(files[0]);
            this.codeDisplay.Text = File.ReadAllText(files[0]);
        }
    }
}
