namespace LostTech.Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using numpy;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;
    using tensorflow.keras.optimizers;

    static class SimpleApproximationProgram {
        const int x0 = 10, x1 = 20;
        const int testSize = 2000;
        const int iterations = 1000;
        const float learningRate = 0.01f;
        const int hiddenSize = 10;

        static void Main() {
            GradientLog.OutputWriter = Console.Out;
            GradientEngine.UseEnvironmentFromVariable();
            var input = tf.keras.Input(new TensorShape(1));

            var hiddenLayer = new Dense(hiddenSize, activation: tf.keras.activations.sigmoid_fn).__call__(input);
            var output = new Dense(1, activation: tf.keras.activations.sigmoid_fn).__call__(hiddenLayer);

            var model = new Model(new {inputs = input, outputs = output}.AsKwArgs());
            model.compile(optimizer: new SGD(learning_rate: learningRate),
                          loss: tf.keras.losses.MSE_fn);

            var (validationInputs, validationOutputs) = GenerateTestValues();

            foreach (int iteration in Enumerable.Range(0, iterations)) {
                var (trainInputs, trainOutputs) = GenerateTestValues();
                model.fit(trainInputs, trainOutputs, batchSize: testSize/50,
                          epochs: iteration+1, stepsPerEpoch: 1, initialEpoch: iteration,
                          validationInput: validationInputs, validationTarget: validationOutputs);
            }
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
