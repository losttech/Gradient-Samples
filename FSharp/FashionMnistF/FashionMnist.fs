open System
open System.Collections.Generic
open System.Runtime.InteropServices
open FSharp.Interop.Dynamic
open FSharp.Interop.Dynamic.Operators

open LostTech.Gradient
open LostTech.TensorFlow
open numpy
open tensorflow
open tensorflow.keras
open tensorflow.keras.layers
open tensorflow.keras.optimizers

// this is no longer needed in most scenarios with 1.15, but we left it for example
let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)
// F# does not use implicit conversions, when resolving an overload
// so this has to be applied explicitly
let inline implicit (x:^a): ImplicitContainer< ^a > = !> x

[<EntryPoint>]
let main argv =
    TensorFlowSetup.Instance.OptInToUsageDataCollection()
    GradientEngine.UseEnvironmentFromVariable() |> ignore

    GradientLog.OutputWriter <- Console.Out

    let struct (train, test) = tf.keras.datasets.fashion_mnist.load_data()
    let trainImages: ndarray = train?Item1 ?/? 255.0f
    let trainLabels = train?Item2
    let testImages: ndarray = test?Item1 ?/? 255.0f
    let testLabels: ndarray = test?Item2

    let model = Sequential([|
        Flatten(kwargs = {| input_shape = struct (28, 28) |}.AsKwArgs());
        Dense(units = 128, activation = tf.nn.relu_fn);
        Dense(units = 10, activation = tf.nn.softmax_fn);
    |]: obj array)

    model.compile(
        optimizer = implicit<obj>(Adam()),
        loss = tf.keras.losses.sparse_categorical_crossentropy_fn,
        metrics = (PyList.ofSeq<obj> ["accuracy"] :> _ seq))

    model.fit(trainImages, trainLabels, epochs = 5) |> ignore

    let evalResult = model.evaluate(testImages, testLabels)
    let accuracy = Core.Operators.float32 (Dyn.getIndexer[1] evalResult  : numpy.float32)
    printfn "Test accuracy: %f" accuracy

    model.summary()

    0 // return an integer exit code
