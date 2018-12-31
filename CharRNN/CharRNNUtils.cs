namespace CharRNN {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using Newtonsoft.Json;
    using numpy;
    using SharPy.Runtime;

    class TextLoader {
        readonly string dataDir;
        readonly int batchSize;
        readonly int seqLength;
        readonly Encoding encoding;
        dynamic tensor;
        internal readonly Dictionary<char, int> vocabulary;
        internal readonly IEnumerable<char> chars;
        internal int vocabularySize;
        internal int batchCount;
        int pointer;
        dynamic x_batches;
        dynamic y_batches;

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
                this.tensor = this.Preprocess(inputFile, vocabularyFile, tensorFile, out this.chars, out this.vocabulary);
            } else {
                Console.WriteLine("loading preprocessed files");
                this.tensor = this.LoadPreprocessed(vocabularyFile, tensorFile, out this.chars, out this.vocabulary);
            }
        }

        ndarray Preprocess(string inputFile, string vocabularyFile, string tensorFile,
            out IEnumerable<char> chars, out Dictionary<char, int> vocabulary) {
            string data = File.ReadAllText(inputFile, this.encoding);
            var counter = Counts(data);
            chars = counter.OrderByDescending(p => p.Value).Select(kv => kv.Key);
            this.vocabularySize = chars.Count();
            vocabulary = this.chars.Select((chr, i) => (chr, i)).ToDictionary(i => i.chr, i => i.i);
            File.WriteAllText(vocabularyFile, JsonConvert.SerializeObject(this.chars));
            var tensor = np.array(data.Select(c => this.vocabulary[c]));
            np.save(tensorFile, tensor);
            return tensor;
        }
        ndarray LoadPreprocessed(string vocabularyFile, string tensorFile,
            out IEnumerable<char> chars, out Dictionary<char, int> vocabulary) {
            chars = JsonConvert.DeserializeObject<IEnumerable<char>>(File.ReadAllText(vocabularyFile));
            this.vocabularySize = this.chars.Count();
            vocabulary = this.chars.Select((chr, i) => (chr, i)).ToDictionary(i => i.chr, i => i.i);
            var tensor = (ndarray)np.load(tensorFile);
            this.batchCount = (int)(tensor.size / (this.batchSize * this.seqLength));
            return tensor;
        }
        void CreateBatches() {
            this.batchCount = (int)(tensor.size / (this.batchSize * this.seqLength));
            if (this.batchCount == 0)
                throw new ArgumentException();

            // TODO implement and use C# 8 indexing here
            this.tensor = this.tensor[..this.batchCount * this.batchSize * this.seqLength];
            _ArrayLike xdata = this.tensor;
            _ArrayLike ydata = np.copy(this.tensor);
            ydata[..-1] = xdata[1..];
            ydata[-1] = xdata[0];
            this.x_batches = np.split((ndarray)xdata.reshape((this.batchSize, -1)), new PythonList<object> { this.batchCount }, 1);
            this.y_batches = np.split((ndarray)ydata.reshape((this.batchSize, -1)), new PythonList<object> { this.batchCount }, 1);
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

        static Dictionary<TValue, int> Counts<TValue>(IEnumerable<TValue> enumerable) {
            var result = new Dictionary<TValue, int>();
            foreach (TValue value in enumerable) {
                result.TryGetValue(value, out int count);
                count++;
                result[value] = count;
            }
            return result;
        }
    }
}
