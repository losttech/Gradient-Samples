namespace CharRNN {
    using System;
    using CommandLine;

    [Verb("train")]
    class CharRNNTrainingParameters: CharRNNModelParameters {
        [Option("log-dir", Default = "logs")]
        public string logDir { get; set; }
        [Option("data-dir", Default = "data/tinyshakespeare")]
        public string dataDir { get; set; }
        [Option("save-dir", Default = "save")]
        public string saveDir { get; set; }
        [Option("save-every", Default = 1000)]
        public int saveEvery { get; set; }
        [Option("init-from")]
        public string initFrom { get; set; }
        [Option("epochs", Default = 50)]
        public int epochs { get; set; }

        [Option("learning-rate", Default = 0.002)]
        public double learningRate { get; set; }
        [Option("decay-rate", Default = 0.97)]
        public  double decayRate { get; set; }

        public CharRNNTrainingParameters() {
            throw new NotImplementedException();
        }
    }
}
