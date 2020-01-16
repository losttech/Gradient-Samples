open FSharp.Interop.Dynamic

open System
open Gradient
open Gradient.BuiltIns
open tensorflow
open tensorflow.summary
open tensorflow.core.protobuf.config_pb2
open tensorflow.python.ops.gen_bitwise_ops

let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)
// F# does not use implicit conversions, when resolving an overload
// so this has to be applied explicitly
let inline implicit (x:^a): ImplicitContainer< ^a > = !> x

[<EntryPoint>]
let main argv =
    GradientSetup.OptInToUsageDataCollection()
    GradientEngine.UseEnvironmentFromVariable() |> ignore

    GradientLog.OutputWriter <- Console.Out

    // Loosing static typing here, because
    // F# does not do perform covariant coversions automatically :(
    // This should be fixed later when types in ops will become well-specified
    let a = tf.constant(5.0, name="a")
    let b = tf.constant(10.0, name="b")

    let sum = tf.add(a, b, name="sum")
    let div = tf.divide(a, b, name="div")

    let x = tf.constant(0b101, name="B101")
    let y = tf.constant(0b010, name="B010")

    let xor = tf.bitwise.bitwise_xor(x, y)
    let bitcount = gen_bitwise_ops.population_count_dyn(xor)

    let config = !? config_pb2.ConfigProto ()
    config?gpu_options?allow_growth <- true

    Session.NewDyn(config=config).UseSelf(fun sess ->
        let writer = FileWriter(".",  sess.graph :?> Graph |> implicit)
        printfn "a = %O" (sess.run a)
        printfn "b = %O" (sess.run b)
        printfn "a + b = %O" (sess.run sum)
        printfn "a / b = %O" (sess.run div)
        printfn ""

        let xorBinary = Convert.ToString(sess.run xor |> Dyn.implicitConvert<int>, toBase=2).PadLeft(3, '0')
        printfn "101 ^ 010 = %s with popcount: %O" xorBinary (sess.run bitcount)
        
        writer.close()
        sess.close())

    0
