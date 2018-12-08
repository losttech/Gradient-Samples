namespace CharRNN {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using Newtonsoft.Json;
    using numpy;

    class TextLoader {
        readonly string dataDir;
        readonly int batchSize;
        readonly int seqLength;
        readonly Encoding encoding;
        readonly dynamic tensor;
        Dictionary<char, int> vocabulary;
        IEnumerable<char> chars;
        internal int vocabularySize;
        internal int batchCount;
        int pointer;

        public TextLoader(string dataDir, int batchSize, int seqLength, Encoding encoding = null) {
            this.dataDir = dataDir;
            this.batchSize = batchSize;
            this.seqLength = seqLength;
            this.encoding = encoding ?? Encoding.UTF8;

            string inputFile = Path.Combine(dataDir, "input.txt");
            string vocabularyFile = Path.Combine(dataDir, "vocab.pkl");
            string tensorFile = Path.Combine(dataDir, "data.npy");

            if (!(File.Exists(vocabularyFile) && File.Exists(tensorFile))) {
                Console.WriteLine("reading text file");
                this.tensor = this.Preprocess(inputFile, vocabularyFile, tensorFile);
            } else {
                Console.WriteLine("loading preprocessed files");
                this.tensor = this.LoadPreprocessed(vocabularyFile, tensorFile);
            }
        }

        ndarray Preprocess(string inputFile, string vocabularyFile, string tensorFile) {
            string data = File.ReadAllText(inputFile, this.encoding);
            var counter = new Dictionary<char, int>(count chars in data);
            this.chars = counter.OrderByDescending(p => p.Value).Select(kv => kv.Key);
            this.vocabularySize = chars.Count();
            this.vocabulary = this.chars.Select((chr, i) => (chr, i)).ToDictionary(i => i.chr, i => i.i);
            File.WriteAllText(vocabularyFile, JsonConvert.SerializeObject(this.chars));
            var tensor = np.array(data.Select(c => this.vocabulary[c]));
            np.save(tensorFile, tensor);
            return tensor;
        }
        ndarray LoadPreprocessed(string vocabularyFile, string tensorFile) {
            this.chars = JsonConvert.DeserializeObject<IEnumerable<char>>(File.ReadAllText(vocabularyFile));
            this.vocabularySize = this.chars.Count();
            this.vocabulary = this.chars.Select((chr, i) => (chr, i)).ToDictionary(i => i.chr, i => i.i);
            var tensor = np.load(tensorFile);
            this.batchCount = (int)(tensor.size / (this.batchSize * this.seqLength));
            return tensor;
        }
        void CreateBatches() {
            this.batchCount = (int)(tensor.size / (this.batchSize * this.seqLength));
            if (this.batchCount == 0)
                throw new ArgumentException();

            // TODO implement and use C# 8 indexing here
            this.tensor = this.tensor[..this.batchCount * this.batchSize * this.seqLength];
            var xdata = this.tensor;
            var ydata = np.copy(this.tensor);
            ydata[..-1] = xdata[1..];
            ydata[-1] = xdata[0];
            this.x_batches = np.split(xdata.reshape(this.batchSize, -1), this.batchCount, 1);
            this.y_batches = np.split(ydata.reshape(this.batchSize, -1), this.batchCount, 1);
        }
        public ValueTuple<dynamic, dynamic> NextBatch() {
            var x = this.x_batches[this.pointer];
            var y = this.y_batches[this.pointer];
            this.pointer++;
            return (x, y);
        }
        internal void ResetBatchPointer() {
            this.pointer = 0;
        }
    }
}
