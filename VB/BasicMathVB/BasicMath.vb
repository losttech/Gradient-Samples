Imports Gradient
Imports SharPy.Runtime
Imports tensorflow
Imports tensorflow.core.protobuf.config_pb2
Imports tensorflow.summary

Module Program
    Sub Main(args As String())
        GradientSetup.OptInToUsageDataCollection()
        GradientSetup.UseEnvironmentFromVariable()

        GradientLog.OutputWriter = Console.Out

        Dim a = tf.constant(5.0, name:="a")
        Dim b = tf.constant(10.0, name:="b")

        Dim sum = tf.add(a, b, name:="sum")
        Dim div = tf.div(a, b, name:="div")

        ' ConfigProto here must be wrapped in () to tell Visual Basic,
        ' that .ConfigProto() is not the same as .ConfigProto.
        ' Alternatively, one can write .ConfigProto()()
        Dim config = (config_pb2.ConfigProto)()
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
                                                       writer.close()
                                                       .close()
                                                   End With
                                               End Sub)
    End Sub
End Module
