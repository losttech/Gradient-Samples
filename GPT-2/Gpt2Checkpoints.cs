namespace Gradient.Samples.GPT2 {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using tensorflow;

    public static class Gpt2Checkpoints {
        public const string CheckpointDir = "checkpoint";

        public static string GetLatestCheckpoint(string gpt2Root, string modelName, string run) {
            string latestCheckpoint = run is null
                ? null
                : tf.train.latest_checkpoint(Path.GetFullPath(Path.Combine(gpt2Root, CheckpointDir, run)));
            latestCheckpoint = latestCheckpoint ?? GetOriginalCheckpoint(gpt2Root, modelName);
            return latestCheckpoint;
        }

        public static string GetOriginalCheckpoint(string gpt2Root, string modelName)
            => tf.train.latest_checkpoint(Path.GetFullPath(Path.Combine(gpt2Root, "models", modelName)));

        public static string ProcessCheckpointConfig(string gpt2Root, string checkpoint,
            string modelName, string runName) {
            switch (checkpoint) {
            case "latest":
                return GetLatestCheckpoint(gpt2Root, modelName, runName);
            case "fresh":
                return GetOriginalCheckpoint(gpt2Root, modelName);
            }

            return checkpoint;
        }
    }
}
