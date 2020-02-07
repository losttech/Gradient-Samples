namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using numpy;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.train;

    static class SimpleApproximationProgram {
        const int x0 = 10, x1 = 20;
        const int testSize = 2000;
        const int iterations = 1000;
        const float learningRate = 0.01f;
        const int hiddenSize = 10;

        static void Main() {
            GradientLog.OutputWriter = Console.Out;
            GradientEngine.UseEnvironmentFromVariable();

            var input = tf.placeholder(tf.float32, new TensorShape(null, 1), name: "x");
            var output = tf.placeholder(tf.float32, new TensorShape(null, 1), name: "y");

            var hiddenLayer = tf.layers.dense(input, hiddenSize,
                activation: tf.sigmoid_fn,
                kernel_initializer: new ones_initializer(),
                bias_initializer: new random_uniform_initializer(minval: -x1, maxval: -x0),
                name: "hidden");

            var model = tf.layers.dense(hiddenLayer, units: 1, name: "output");

            var cost = tf.losses.mean_squared_error(output, model);

            var training = new GradientDescentOptimizer(learning_rate: learningRate).minimize(cost);

            dynamic init = tf.global_variables_initializer();

            new Session().UseSelf(session => {
                session.run(new[] { init });

                foreach (int iteration in Enumerable.Range(0, iterations)) {
                    var (trainInputs, trainOutputs) = GenerateTestValues();
                    var iterationDataset = new PythonDict<dynamic, object> {
                        [input] = trainInputs,
                        [output] = trainOutputs,
                    };
                    session.run(new[] { training }, feed_dict: iterationDataset);

                    if (iteration % 100 == 99)
                        Console.WriteLine($"cost = {session.run(new[] { cost }, feed_dict: iterationDataset)}");
                }

                var (testInputs, testOutputs) = GenerateTestValues();

                var testValues = session.run(new[] { model }, feed_dict: new PythonDict<dynamic, object> {
                    [input] = testInputs,
                });

                new variable_scope("hidden", reuse: true).UseSelf(_ => {
                    Variable w = tf.get_variable("kernel");
                    Variable b = tf.get_variable("bias");
                    Console.WriteLine("hidden:");
                    Console.WriteLine($"kernel= {w.eval()}");
                    Console.WriteLine($"bias  = {b.eval()}");
                });

                new variable_scope("output", reuse: true).UseSelf(_ => {
                    Variable w = tf.get_variable("kernel");
                    Variable b = tf.get_variable("bias");
                    Console.WriteLine("hidden:");
                    Console.WriteLine($"kernel= {w.eval()}");
                    Console.WriteLine($"bias  = {b.eval()}");
                });
            });
        }

        static (ndarray, ndarray) GenerateTestValues() {
            double Fun(double input) => Math.Sin(input);

            var inputs = new List<double>();
            var outputs = new List<double>();

            var random = new Random();
            foreach(int _ in Enumerable.Range(0, testSize)) {
                double x = x0 + (x1 - x0) * random.NextDouble();
                double y = Fun(x);
                inputs.Add(x);
                outputs.Add(y);
            }

            return (
                (ndarray)inputs.ToNumPyArray().reshape(new[] { inputs.Count, 1 }),
                (ndarray)outputs.ToNumPyArray().reshape(new[] { outputs.Count, 1 }));
        }
    }
}
