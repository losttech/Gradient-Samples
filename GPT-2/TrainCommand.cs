namespace Gradient.Samples.GPT2 {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using ManyConsole.CommandLineUtils;
    using numpy;
    using Python.Runtime;
    using DataSet = System.Collections.Generic.List<numpy.ndarray>;
    class TrainCommand: ConsoleCommand {
        public override int Run(string[] remainingArguments) {
            this.CheckRequiredArguments();
            if (remainingArguments.Length < 1)
                throw new ArgumentNullException("dataset");
            string datasetName = remainingArguments[0];
            string checkpoint = Gpt2Checkpoints.ProcessCheckpointConfig(
                gpt2Root: Environment.CurrentDirectory,
                checkpoint: this.Checkpoint,
                modelName: this.ModelName,
                runName: this.RunName);

            var encoder = Gpt2Encoder.LoadEncoder(this.ModelName);
            string searchPattern = this.Include ?? "*";
            var dataset = searchPattern.EndsWith("*.csv")
                ? LoadCsv(encoder, root: datasetName, field: this.ColumnName)
                : Gpt2Dataset.LoadDataset(encoder, path: datasetName, pattern: searchPattern);
            if (dataset.Count == 0) {
                Console.Error.WriteLine("The dataset is empty!");
                return -1;
            }
            var hParams = Gpt2Model.LoadHParams(this.ModelName);
            var random = this.Seed == null ? new Random() : new Random(this.Seed.Value);
            var stop = new CancellationTokenSource();
            Console.CancelKeyPress += delegate { stop.Cancel(); };
            new Gpt2Trainer(dataset, encoder, hParams, this.BatchSize, this.SampleLength, random)
                .Train(checkpoint, this.RunName, stop.Token);

            return 0;
        }

        static DataSet LoadCsv(Gpt2Encoder encoder, string root, string field) {
            var result = new List<string>();
            foreach (string file in Directory
                    .EnumerateFiles(root, "*.csv", SearchOption.AllDirectories)) {
                using (var reader = new CsvHelper.CsvReader(new StreamReader(file, Encoding.UTF8),
                    new CsvHelper.Configuration.Configuration {
                        Delimiter = ",",
                        HasHeaderRecord = true,
                    })) {
                    reader.Read();
                    reader.ReadHeader();
                    while (reader.Read()) {
                        string entry = reader.GetField(field);
                        System.Diagnostics.Debug.Assert(reader.GetField(0).Length < 300);
                        if (!string.IsNullOrWhiteSpace(entry))
                            result.Add(entry);
                    }
                }
            }
            return Load(encoder, result);
        }

        const int TrimAfter = 16 * 1024 * 1024;

        static DataSet Load(Gpt2Encoder encoder, IEnumerable<string> texts) {
            dynamic numpy = Py.Import("numpy");
            var result = new DataSet();
            string endOfText = Gpt2Encoder.EndOfTextPseudoToken;
            var chunk = new List<string>();
            int chunkSize = 0;
            void AddChunk() {
                PyObject tokens = numpy.stack(chunk);
                chunk.Clear();
                chunkSize = 0;
                result.Add(ndarray.Wrap(tokens));
            }
            foreach (string text in texts) {
                if (string.IsNullOrWhiteSpace(text))
                    continue;

                if (chunkSize + text.Length + endOfText.Length >= TrimAfter) {
                    AddChunk();
                } else {
                    chunkSize += text.Length + endOfText.Length;
                    var encoded = encoder.Encode(text);
                    chunk.AddRange(encoded);
                    chunk.Add(endOfText);
                }
            }
            if (chunk.Count > 0)
                AddChunk();
            return result;
        }

        public string ModelName { get; set; } = "117M";
        public int? Seed { get; set; }
        public int BatchSize { get; set; } = 1;
        public int SampleLength { get; set; } = 1024;
        public int SampleNum { get; set; } = 1;
        public int SampleEvery { get; set; } = 100;
        public int SaveEvery { get; set; } = 1000;
        public string RunName { get; set; } = DateTime.Now.ToString("s").Replace(':', '-');
        public string Checkpoint { get; set; } = "latest";
        public string Include { get; set; }
        public string ColumnName { get; set; }

        public TrainCommand() {
            this.IsCommand("train");
            this.HasAdditionalArguments(1, "<dataset>");
            this.HasOption("m|model=", "Which model to use", name => this.ModelName = name);
            this.HasOption("s|seed=",
                "Explicitly set seed for random generators to get reproducible results",
                (int s) => this.Seed = s);
            this.HasOption("i|include=", "Pattern of files to include in training",
                pattern => this.Include = pattern);
            this.HasOption("n|sample-num=", "",
                (int count) => this.SampleNum = count);
            this.HasOption("b|batch-size=", "Size of the batch, must divide sample-count",
                (int size) => this.BatchSize = size);
            this.HasOption("l|sample-length=", "",
                (int len) => this.SampleLength = len);
            this.HasOption("sample-every=", "Print a sample every N epochs",
                (int n) => this.SampleEvery = n);
            this.HasOption("save-every=", "How often to save a model, in epochs",
                (int n) => this.SaveEvery = n);
            this.HasOption("r|run=", "Name of the run (to be able to resume)",
                run => this.RunName = run);
            this.HasOption("c|checkpoint=", "Use specific checkpoint to start. Available values: 'latest' (default), 'fresh', or path to a checkpoint file",
                checkpoint => this.Checkpoint = checkpoint);
            this.HasOption("column=", "Read texts from specific CSV column",
                name => this.ColumnName = name);
        }
    }
}
