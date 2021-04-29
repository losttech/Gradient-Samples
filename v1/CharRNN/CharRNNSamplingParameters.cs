namespace CharRNN {
    using System;
    using CommandLine;
    [Verb("sample")]
    class CharRNNSamplingParameters {
        [Option("save-dir", Default = "save")]
        public string saveDir { get; set; }
        [Option("prime", Default = "")]
        public string prime { get; set; }
        [Option("count", Default = 200)]
        public int count { get; set; }
    }
}
