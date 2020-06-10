namespace LinearSVM {
    using System;
    using System.Collections;
    using System.Diagnostics;
    using System.Dynamic;
    using System.Linq;
    using LostTech.Gradient;
    using LostTech.Gradient.ManualWrappers;
    using ManyConsole.CommandLineUtils;
    using numpy;
    using Python.Runtime;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.train;

    class LinearSvmProgram {
        static readonly Random random = new Random();

        readonly LinearSvmCommand flags;

        public LinearSvmProgram(LinearSvmCommand flags)
        {
            this.flags = flags ?? throw new ArgumentNullException(nameof(flags));
        }

        static int Main(string[] args) {
            Console.Title = "LinearSVM";
            GradientLog.OutputWriter = Console.Out;
            GradientEngine.UseEnvironmentFromVariable();

            // required before using PythonEngine
            GradientSetup.EnsureInitialized();
            return ConsoleCommandDispatcher.DispatchCommand(
               ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(LinearSvmProgram)),
               args, Console.Out);
        }

        public int Run()
        {
            dynamic datasets = Py.Import("sklearn.datasets");
            dynamic slice = PythonEngine.Eval("slice");
            var iris = datasets.load_iris();
            dynamic firstTwoFeaturesIndex = new PyTuple(new PyObject[] {
                slice(null),
                slice(null, 2)
            });
            var input = iris.data.__getitem__(firstTwoFeaturesIndex);
            IEnumerable target = iris.target;
            var expectedOutput = target.Cast<dynamic>()
                .Select(l => (int)l == 0 ? 1 : -1)
                .ToArray();
            int trainCount = expectedOutput.Length * 4 / 5;
            var trainIn = np.array(((IEnumerable)input).Cast<dynamic>().Take(trainCount));
            var trainOut = np.array(expectedOutput.Take(trainCount));
            var testIn = np.array(((IEnumerable)input).Cast<dynamic>().Skip(trainCount));
            var testOut = np.array(expectedOutput.Skip(trainCount));

            var inPlace = tf.placeholder(shape: new TensorShape(null, input.shape[1]), dtype: tf.float32);
            var outPlace = tf.placeholder(shape: new TensorShape(null, 1), dtype: tf.float32);
            var w = new Variable(tf.random_normal(shape: new TensorShape((int)input.shape[1], 1)));
            var b = new Variable(tf.random_normal(shape: new TensorShape(1, 1)));

            var totalLoss = Loss(w, b, inPlace, outPlace);
            var accuracy = Inference(w, b, inPlace, outPlace);

            var trainOp = new GradientDescentOptimizer(this.flags.InitialLearningRate).minimize(totalLoss);

            var expectedTrainOut = trainOut.reshape(new int[] { trainOut.Length, 1 });
            var expectedTestOut = testOut.reshape(new int[] { testOut.Length, 1 });

            new Session().UseSelf(sess =>
            {
                var init = tf.global_variables_initializer();
                sess.run(init);
                for(int step = 0; step < this.flags.StepCount; step++)
                {
                    (ndarray @in, ndarray @out) = this.NextBatch(trainIn, trainOut, sampleCount: this.flags.BatchSize);
                    var feed = new PythonDict<object, object> {
                        [inPlace] = @in,
                        [outPlace] = @out,
                    };
                    sess.run(trainOp, feed_dict: feed);

                    var loss = sess.run(totalLoss, feed_dict: feed);
                    var trainAcc = sess.run(accuracy, new PythonDict<object, object>
                    {
                        [inPlace] = trainIn,
                        [outPlace] = expectedTrainOut,
                    });
                    var testAcc = sess.run(accuracy, new PythonDict<object, object>
                    {
                        [inPlace] = testIn,
                        [outPlace] = expectedTestOut,
                    });

                    if ((step + 1) % 100 == 0)
                        Console.WriteLine($"Step{step}: test acc {testAcc}, train acc {trainAcc}");
                }
            });

            return 0;
        }

        dynamic Loss(dynamic W, dynamic b, dynamic inputData, dynamic targetData) {
            var logits = tf.subtract(tf.matmul(inputData, W), b);
            var normTerm = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W), W)), 2);
            var classificationLoss = tf.reduce_mean(tf.maximum(tf.constant(0.0), tf.subtract(this.flags.Delta, tf.multiply(logits, targetData))));
            var totalLoss = tf.add_dyn(tf.multiply(this.flags.C, classificationLoss), tf.multiply(this.flags.Reg, normTerm));
            return totalLoss;
        }

        static dynamic Inference(IGraphNodeBase W, IGraphNodeBase b, dynamic inputData, dynamic targetData) {
            var prediction = tf.sign_dyn(tf.subtract(tf.matmul(inputData, W), b));
            var accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, targetData), tf.float32));
            return accuracy;
        }

        (ndarray, ndarray) NextBatch(dynamic inputData, ndarray targetData, int? sampleCount = null) {
            sampleCount ??= this.flags.BatchSize;
            int max = inputData.Length;
            var indexes = Enumerable.Range(0, sampleCount.Value)
                .Select(_ => random.Next(max))
                .ToArray();

            ndarray inputBatch = inputData[indexes];
            var outputBatch = (ndarray)targetData[indexes].reshape((sampleCount.Value, 1).Items());
            if (outputBatch is null)
                throw new InvalidOperationException();
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
