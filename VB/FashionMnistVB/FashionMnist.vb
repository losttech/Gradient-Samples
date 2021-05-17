Imports System
Imports tensorflow
Imports tensorflow.keras
Imports tensorflow.keras.layers
Imports tensorflow.keras.optimizers

Module Program
    Sub Main()
        GradientLog.OutputWriter = Console.Out
        GradientEngine.UseEnvironmentFromVariable()

        ' requires Internet connection
        Dim data = tf.keras.datasets.fashion_mnist.load_data()
        Dim train = data.Item1
        Dim trainImages = train.Item1 / 255.0F
        Dim trainLabels = train.Item2
        Dim test = data.Item2
        Dim testImages = test.Item1 / 255.0F
        Dim testLabels = test.Item2

        Dim model = New Sequential(New Layer() {
            New Flatten(kwargs:=New With {.input_shape = (28, 28)}.AsKwArgs),
            New Dense(units:=128, activation:=tf.nn.leaky_relu_fn),
            New Dense(units:=10, activation:=tf.keras.activations.softmax_fn)
        })

        model.compile(
            optimizer:=New ImplicitContainer(Of Object)(New Adam()),
            loss:="sparse_categorical_crossentropy",
            metrics:=New String() {"accuracy"})

        model.fit(trainImages, trainLabels, epochs:=5)

        Dim evaluationResult = model.evaluate(testImages, testLabels)
        Dim accuracy = evaluationResult(1)
        Console.WriteLine($"Test accuracy: {accuracy}")

        model.summary()
    End Sub
End Module
