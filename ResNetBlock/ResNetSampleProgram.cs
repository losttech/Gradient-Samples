﻿namespace Gradient.Samples {
    using System;
    using System.Diagnostics;
    using numpy;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;
    using tensorflow.train;

    static class ResNetSampleProgram {
        public static void Run() {
            var block = new ResNetBlock(kernelSize: 1, filters: new[] { 1, 2, 3 });
            var applicationResult = block.call(tf.zeros(new TensorShape(1, 2, 3, 3)));
            Console.WriteLine(applicationResult);

            foreach (dynamic variable in block.trainable_variables) {
                Console.WriteLine(variable.name);
            }

            // requires Internet connection
            (dynamic train, dynamic test) = tf.keras.datasets.fashion_mnist.load_data();
            // will be able to do (trainImages, trainLabels) = train;
            ndarray trainImages = train.Item1 / 255.0f;
            ndarray trainLabels = train.Item2;
            ndarray testImages = test.Item1 / 255.0f;
            ndarray testLabels = test.Item2;

            bool loaded = 60000 == trainImages.Length;
            Debug.Assert(loaded);

            var model = new Sequential(new Layer[] {
                // will be able to do: new Flatten(kwargs: new { input_shape = (28, 28) }),
                new Flatten(kwargs: new PythonDict<string, object> { ["input_shape"] = (28, 28) }),
                new ResNetBlock(kernelSize: 3, filters: new [] { 1, 2, 3 }),
                new ResNetBlock(kernelSize: 3, filters: new [] { 1, 2, 3 }),
                new Dense(units: 10, activation: tf.nn.softmax_fn),
            });

            model.compile(
                optimizer: new AdamOptimizer(),
                loss: "sparse_categorical_crossentropy",
                metrics: new dynamic[] { "accuracy" });

            model.fit(trainImages, trainLabels, epochs: 5);

            var testEvalResult = model.evaluate(testImages, testLabels);
            double testAcc = testEvalResult[1];

            Console.WriteLine($"Test accuracy: {testAcc}");
        }
        static void Main() {
            Console.Title = nameof(ResNetSampleProgram);
            GradientLog.OutputWriter = Console.Out;
            GradientSetup.UseEnvironmentFromVariable();
            Run();
        }
    }
}