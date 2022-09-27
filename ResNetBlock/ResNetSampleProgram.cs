namespace LostTech.Gradient.Samples {
    using System;
    using System.Diagnostics;
    using numpy;

    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;
    using tensorflow.keras.optimizers;

    static class ResNetSampleProgram {
        public static void Run(int epochs = 5) {
            // requires Internet connection
            (dynamic train, dynamic test) = tf.keras.datasets.fashion_mnist.load_data();
            ndarray trainImages = np.expand_dims(train[0] / 255.0f, axis: 3);
            ndarray trainLabels = train[1];
            ndarray testImages = np.expand_dims(test[0] / 255.0f, axis: 3);
            ndarray testLabels = test[1];

            bool loaded = 60000 == trainImages.Length;
            Debug.Assert(loaded);

            var model = new Sequential(new Layer[] {
                new InputLayer(new TensorShape(28, 28, 1)),
                new ResNetBlock(kernelSize: 3, filters: new [] { 1, 2, 3 }),
                new ResNetBlock(kernelSize: 3, filters: new [] { 1, 2, 3 }),
                new Flatten(),
                new Dense(units: 10, activation: tf.keras.activations.softmax_fn),
            });

            model.compile(
                optimizer: new Adam(),
                loss: "sparse_categorical_crossentropy",
                metrics: new [] { "accuracy" });

            model.fit(trainImages, trainLabels, epochs: epochs);

            var testEvalResult = model.evaluate(testImages, testLabels);
            double testAcc = testEvalResult[1];

            Console.WriteLine($"Test accuracy: {testAcc}");
            model.summary();
        }

        static void Main() {
            Console.Title = nameof(ResNetSampleProgram);
            GradientLog.OutputWriter = Console.Out;
            GradientEngine.UseEnvironmentFromVariable();
            Run();
        }
    }
}
