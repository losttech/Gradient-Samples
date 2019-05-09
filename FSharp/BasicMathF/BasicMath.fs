open FSharp.Interop.Dynamic

open System
open Gradient
open tensorflow
open tensorflow.summary

let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)
// F# does not use implicit conversions, when resolving an overload
// so this has to be applied explicitly
let inline implicit (x:^a): SharPy.Runtime.ImplicitContainer< ^a > = !> x

[<EntryPoint>]
let main argv =
    GradientSetup.OptInToUsageDataCollection()

    GradientLog.OutputWriter <- Console.Out

    // Loosing static typing here, because
    // F# does not do perform covariant coversions automatically :(
    // This should be fixed later when types in ops will become well-specified
    let a = tf.constant(5.0, name="a") |> Seq.singleton<obj>
    let b = tf.constant(10.0, name="b") |> Seq.singleton<obj>

    let sum = tf.add(a, b, name="sum")
    let div = tf.div(a, b, name="div") |> Seq.singleton<obj>

    Session().UseSelf(fun sess ->
        let writer = FileWriter(".",  sess.graph :?> Graph |> implicit)
        printfn "a = %O" (sess.run a)
        printfn "b = %O" (sess.run b)
        printfn "a + b = %O" (sess?run(sum))
        printfn "a / b = %O" (sess.run div)
        
        writer.close()
        sess.close())

    0
