open System
open FSharp.Interop.Dynamic
open FSharp.Interop.Dynamic.Operators

open Gradient
open numpy
open tensorflow
open tensorflow.keras
open tensorflow.keras.layers
open tensorflow.keras.optimizers
open SharPy.Runtime

[<EntryPoint>]
let main argv =
    GradientSetup.OptInToUsageDataCollection()
    GradientSetup.UseEnvironmentFromVariable() |> ignore

    GradientLog.OutputWriter <- Console.Out

    let struct (train, test) = tf.keras.datasets.fashion_mnist.load_data()
    let trainImages: ndarray = train?Item1 ?/? 255.0f
    let trainLabels = train?Item2
    let testImages: ndarray = test?Item1 ?/? 255.0f
    let testLabels: ndarray = test?Item2

    let inputArgs = PythonDict<_,obj>()
    inputArgs.["input_shape"] <- struct (28, 28)
    let model = Sequential([|
        Flatten(kwargs = inputArgs) :> obj;
        Dense(units = 128, activation = tf.nn.relu_fn) :> obj;
        Dense(units = 10, activation = tf.nn.softmax_fn) :> obj;
    |])

    model.compile(
        optimizer = Adam(),
        loss = tf.keras.losses.sparse_categorical_crossentropy_fn,
        metrics = ["accuracy"])

    model.fit(trainImages, trainLabels, epochs = 5) |> ignore

    let evalResult = model.evaluate(testImages, testLabels)
    let accuracy = Core.Operators.float (Dyn.getIndex evalResult [1] : numpy.float64)
    printfn "Test accuracy: %f" accuracy

    model.summary()

    0 // return an integer exit code
