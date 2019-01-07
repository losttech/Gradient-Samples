namespace LinearSVM {
    using System;
    using System.Dynamic;
    using System.Linq;
    using Gradient;
    using Python.Runtime;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.flags;

    static class LinearSvmProgram {
        static dynamic FLAGS;
        static readonly Random random = new Random();
        static dynamic np;

        static void Main() {
            GradientLog.OutputWriter = Console.Out;

            np = PythonEngine.ImportModule("numpy");
            // WIP
            // flags do not work. See https://github.com/pythonnet/pythonnet/issues/792
            //flags.DEFINE_integer("batch_size", 32, "Number of samples per batch");
            //flags.DEFINE_integer("num_steps", 500, "Number of training steps.");
            //flags.DEFINE_boolean("is_evaluation", true, "Whether or not the model should be evaluated.");

            //flags.DEFINE_float("C_param", 0.1, "penalty parameter of the error term.");
            //flags.DEFINE_float(
            //    "Reg_param", 1.0,
            //    "penalty parameter of the error term.");
            //flags.DEFINE_float(
            //    "delta", 1.0,
            //    "The parameter set for margin.");
            //flags.DEFINE_float(
            //    "initial_learning_rate", 0.1,
            //    "The initial learning rate for optimization.");

            //FLAGS = flags.FLAGS;

            FLAGS = new {
                batch_size = 32,
                num_steps = 500,
                is_evaluation = true,
                C_param = 0.1,
                Reg_param = 1.0,
                delta = 1.0,
                initial_learning_rate = 0.1,
            };
        }

        static dynamic Loss(dynamic W, dynamic b, dynamic inputData, dynamic targetData) {
            var logits = tf.subtract(tf.matmul(inputData, W), b);
            var normTerm = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W), W)), 2);
            var classificationLoss = tf.reduce_mean(tf.maximum(0.0, tf.subtract(FLAGS.delta, tf.multiply(logits, targetData))));
            var totalLoss = tf.add(tf.multiply(FLAGS.C_param, classificationLoss), tf.multiply(FLAGS.Reg_param, normTerm));
            return totalLoss;
        }

        static dynamic Inference(dynamic W, dynamic b, dynamic inputData, dynamic targetData) {
            var prediction = tf.sign(tf.subtract(tf.matmul(inputData, W), b));
            var accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, targetData), tf.float32));
            return accuracy;
        }

        static dynamic NextBatch(dynamic inputData, dynamic targetData, int? sampleCount = null) {
            sampleCount = sampleCount ?? FLAGS.batch_size;
            int max = inputData.Length;
            var indexes = Enumerable.Range(0, sampleCount.Value)
                .Select(_ => random.Next(max))
                .ToArray();

            var inputBatch = inputData[indexes];
            var outputBatch = np.transpose(new[] { targetData[indexes] });
            return (inputBatch, outputBatch);
        }

        class ContextManager {
            public static implicit operator ExitAction(ContextManager _) => new ExitAction();
        }

        public struct ExitAction : IDisposable
        {
            public ExitAction(Action onDispose) {
                this.OnDispose = onDispose ?? throw new ArgumentNullException(nameof(onDispose));
            }
            public Action OnDispose { get; }
            public void Dispose() => this.OnDispose?.Invoke();
        }
    }
}
