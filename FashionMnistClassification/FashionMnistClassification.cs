namespace LostTech.Gradient.Samples {
    using System;
    using System.Diagnostics;
    using numpy;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;
    using tensorflow.train;

    static class FashionMnistClassification {
        static void Main() {
            Console.Title = nameof(FashionMnistClassification);
            GradientLog.OutputWriter = Console.Out;
            GradientEngine.UseEnvironmentFromVariable();

            // requires Internet connection
            (dynamic train, dynamic test) = tf.keras.datasets.fashion_mnist.load_data();
            ndarray trainImages = train.Item1 / 255.0f;
            ndarray trainLabels = train.Item2;
            ndarray testImages = test.Item1 / 255.0f;
            ndarray testLabels = test.Item2;

            bool loaded = 60000 == trainImages.Length;
            Debug.Assert(loaded);

            var model = new Sequential(new Layer[] {
                new Flatten(kwargs: new { input_shape = (28, 28) }.AsKwArgs()),
                new Dense(units: 128, activation: tf.nn.selu_fn),
                new Dense(units: 10, activation: tf.nn.softmax_fn),
            });

            model.compile(
                optimizer: new AdamOptimizer(),
                loss: "sparse_categorical_crossentropy",
                metrics: new [] {"accuracy"});

            model.fit(trainImages, trainLabels, epochs: 5);

            var testEvalResult = model.evaluate(testImages, testLabels);
            double testAcc = testEvalResult[1];

            Console.WriteLine($"Test accuracy: {testAcc}");
        }
    }
}
