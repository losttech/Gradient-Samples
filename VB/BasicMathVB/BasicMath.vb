Imports Gradient
Imports SharPy.Runtime
Imports tensorflow
Imports tensorflow.core.protobuf.config_pb2
Imports tensorflow.python.ops.gen_bitwise_ops
Imports tensorflow.summary

Module Program
    Sub Main(args As String())
        GradientSetup.OptInToUsageDataCollection()
        GradientEngine.UseEnvironmentFromVariable()

        GradientLog.OutputWriter = Console.Out

        Dim a = tf.constant(5.0, name:="a")
        Dim b = tf.constant(10.0, name:="b")

        Dim sum = tf.add(a, b, name:="sum")
        Dim div = tf.divide(a, b, name:="div")

        Dim x = tf.constant(&B101, name:="B101")
        Dim y = tf.constant(&B110, name:="B110")

        Dim x_or = tf.bitwise.bitwise_xor(x, y)
        Dim bitcount = gen_bitwise_ops.population_count_dyn(x_or)

        ' ConfigProto here must be wrapped in () to tell Visual Basic,
        ' that .ConfigProto() is not the same as .ConfigProto.
        ' Alternatively, one can write .ConfigProto()()
        Dim config = config_pb2.ConfigProto.CreateInstance()
        config.gpu_options.allow_growth = True
        Session.NewDyn(config:=config).UseSelf(Sub(sess)
                                                   With sess
                                                       ' Visual Basic does not perform implicit conversion from Graph
                                                       ' to ImplicitContainer(Of Graph) :(
                                                       Dim writer = New FileWriter(".", New ImplicitContainer(Of Graph)(.graph))
                                                       Console.WriteLine($"a = { .run(a) }")
                                                       Console.WriteLine($"b = { .run(b) }")
                                                       Console.WriteLine($"a + b = { .run(sum) }")
                                                       Console.WriteLine($"a / b = { .run(div) }")
                                                       Console.WriteLine()

                                                       Dim xorBinary = Convert.ToString(.run(x_or), toBase:=2).PadLeft(3, "0"c)
                                                       Console.WriteLine($"101 ^ 110 = {xorBinary} with popcount: { .run(bitcount)}")

                                                       writer.close()
                                                       .close()
                                                   End With
                                               End Sub)
    End Sub
End Module
