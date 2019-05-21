namespace BasicMath {
    using System;
    using Gradient;
    using tensorflow;
    using tensorflow.core.protobuf.config_pb2;
    using tensorflow.summary;

    static class BasicMathProgram {
        static void Main() {
            GradientLog.OutputWriter = Console.Out;
            GradientSetup.UseEnvironmentFromVariable();

            Tensor a = tf.constant(5.0, name: "a");
            Tensor b = tf.constant(10.0, name: "b");

            Tensor sum = tf.add(a, b, name: "sum");
            Tensor div = tf.div(a, b, name: "div");

            dynamic config = config_pb2.ConfigProto();
            // unless this is set, tensorflow-gpu consumes all of GPU memory
            // don't set it if you don't want you training to crash due to random OOM in the middle
            config.gpu_options.allow_growth = true;
            Session sess = Session.NewDyn(config: config);
            sess.UseSelf(session => {
                var writer = new FileWriter(".", session.graph);
                Console.WriteLine($"a = {session.run(a)}");
                Console.WriteLine($"b = {session.run(b)}");
                Console.WriteLine($"a + b = {session.run(sum)}");
                Console.WriteLine($"a / b = {session.run(div)}");
                writer.close();
                session.close();
            });
        }
    }
}
