namespace Gradient.Samples.GPT2 {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using numpy;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.contrib.training;
    using static System.FormattableString;

    class TrainingSamplePrinter {
        const string SampleDir = "samples";

        readonly IGpt2Decoder<string> decoder;
        readonly int batchSize;
        public int NumberOfSamplesToPrint { get; set; } = 1;
        public int SampleEvery { get; set; } = 100;
        readonly string sampleStartToken;
        readonly int sampleLength;
        readonly HParams hParams;
        dynamic sampler;

        public void GenerateSamples(int epoch, string run, Session session, dynamic context) {
            if (epoch % this.SampleEvery != 0)
                return;

            this.sampler = this.sampler ?? Gpt2Sampler.SampleSequence(
                this.hParams,
                length: this.sampleLength,
                context: context,
                batchSize: this.batchSize,
                temperature: 1.0f,
                topK: 40);
            var contextTokens = np.array(new[] { this.sampleStartToken });
            var allText = new List<string>();
            int index = 0;
            string text = null;
            while (index < this.NumberOfSamplesToPrint) {
                var @out = session.run(this.sampler, feed_dict: new PythonDict<object, object> { [context] = Enumerable.Repeat(contextTokens, this.batchSize), });
                foreach (int i in Enumerable.Range(0, Math.Min(this.NumberOfSamplesToPrint - index, this.batchSize))) {
                    text = this.decoder.Decode(@out[i]);
                    text = Invariant($"======== SAMPLE {index + 1} ========\n{text}\n");
                    allText.Add(text);
                    index++;
                }
            }

            Debug.WriteLine(text);
            string runSampleDir = Path.Combine(SampleDir, run);
            Directory.CreateDirectory(runSampleDir);
            File.WriteAllLines(path: Path.Combine(runSampleDir, Invariant($"samples-{epoch}")), contents: allText);
        }

        public TrainingSamplePrinter(HParams hParams, int batchSize, IGpt2Decoder<string> decoder, string sampleStartToken, int sampleLength) {
            this.hParams = hParams;
            this.batchSize = batchSize;
            this.decoder = decoder;
            this.sampleStartToken = sampleStartToken;
            this.sampleLength = sampleLength;
        }
    }
}
