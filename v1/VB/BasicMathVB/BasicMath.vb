Imports LostTech.Gradient
Imports LostTech.TensorFlow
Imports tensorflow
Imports tensorflow.compat.v1
Imports tensorflow.compat.v1.summary
Imports tensorflow.core.protobuf.config_pb2
Imports tensorflow.python.ops.gen_bitwise_ops

Module Program
    Sub Main(args As String())
        TensorFlowSetup.Instance.OptInToUsageDataCollection()
        GradientEngine.UseEnvironmentFromVariable()

        GradientLog.OutputWriter = Console.Out

        Dim config = config_pb2.ConfigProto.CreateInstance()
        config.gpu_options.allow_growth = True
        Dim sess = Session.NewDyn(config:=config)
        With sess
            Using .StartUsing()
                Dim a = tf.constant(5.0, name:="a")
                Dim b = tf.constant(10.0, name:="b")

                Dim sum = tf.add(a, b, name:="sum")
                Dim div = tf.divide(a, b, name:="div")

                Dim x = tf.constant(&B101, name:="B101")
                Dim y = tf.constant(&B110, name:="B110")

                Dim x_or = tf.bitwise.bitwise_xor(x, y)
                Dim bitcount = gen_bitwise_ops.population_count(x_or)

                Dim writer = New FileWriter(".", CType(.graph, Graph))
                Console.WriteLine($"a = { .run(a) }")
                Console.WriteLine($"b = { .run(b) }")
                Console.WriteLine($"a + b = { .run(sum) }")
                Console.WriteLine($"a / b = { .run(div) }")
                Console.WriteLine()

                Dim xorBinary = Convert.ToString(.run(x_or), toBase:=2).PadLeft(3, "0"c)
                Console.WriteLine($"101 ^ 110 = {xorBinary} with popcount: { .run(bitcount)}")

                writer.close()
            End Using
        End With
    End Sub
End Module
