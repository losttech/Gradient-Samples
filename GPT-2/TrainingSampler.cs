namespace Gradient.Samples.GPT2 {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using numpy;

    class TrainingSampler {
        readonly List<ndarray> chunks;
        readonly List<int> boundaries = new List<int> { 0 };
        readonly Random random;
        public int TokenCount { get; }

        public TrainingSampler(List<ndarray> chunks, Random random) {
            this.random = random ?? throw new ArgumentNullException(nameof(random));
            this.chunks = chunks ?? throw new ArgumentNullException(nameof(chunks));
            this.TokenCount = chunks.Sum(chunk => chunk.shape.Item1);
            if (this.TokenCount == 0)
                throw new ArgumentException("Dataset is empty", paramName: nameof(chunks));

            foreach (var chunk in chunks)
                this.boundaries.Add(this.boundaries[this.boundaries.Count - 1] + chunk.shape.Item1);
        }

        public ndarray Sample(int length) {
            if (length >= this.TokenCount / this.chunks.Count)
                throw new ArgumentException($"Dataset files are too small to sample {length} tokens at a time." +
                                            $"Maximum is {this.TokenCount / this.chunks.Count}.");

            while (true) {
                int index = this.random.Next(this.TokenCount - length);
                int i = Algo.BinarySearch(j => this.boundaries[j] > index,
                            lo: 0, hi: this.boundaries.Count - 1) - 1;

                if (this.boundaries[i + 1] > index + length) {
                    int withinChunk = index - this.boundaries[i];
                    dynamic chunk = this.chunks[i];
                    return chunk[new Range(withinChunk, withinChunk + length - 1)];
                }
            }
        }
    }
}