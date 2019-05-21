open FSharp.Interop.Dynamic

open System
open Gradient
open tensorflow
open tensorflow.summary
open tensorflow.core.protobuf.config_pb2

let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)
// F# does not use implicit conversions, when resolving an overload
// so this has to be applied explicitly
let inline implicit (x:^a): SharPy.Runtime.ImplicitContainer< ^a > = !> x

[<EntryPoint>]
let main argv =
    GradientSetup.OptInToUsageDataCollection()
    GradientSetup.UseEnvironmentFromVariable() |> ignore

    GradientLog.OutputWriter <- Console.Out

    // Loosing static typing here, because
    // F# does not do perform covariant coversions automatically :(
    // This should be fixed later when types in ops will become well-specified
    let a = tf.constant(5.0, name="a")
    let b = tf.constant(10.0, name="b")

    let sum = tf.add(a, b, name="sum")
    let div = tf.div(a, b, name="div")

    let config = !? config_pb2.ConfigProto ()
    config?gpu_options?allow_growth <- true

    Session.NewDyn(config=config).UseSelf(fun sess ->
        let writer = FileWriter(".",  sess.graph :?> Graph |> implicit)
        printfn "a = %O" (sess.run a)
        printfn "b = %O" (sess.run b)
        printfn "a + b = %O" (sess.run sum)
        printfn "a / b = %O" (sess.run div)
        
        writer.close()
        sess.close())

    0
