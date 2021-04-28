namespace LostTech.Gradient.Samples {
    using System;
    using LostTech.Gradient;
    using tensorflow;
    using tensorflow.compat.v1;
    using tensorflow.compat.v1.summary;
    using tensorflow.core.protobuf.config_pb2;
    using tensorflow.python.ops.gen_bitwise_ops;

    static class BasicMathProgram {
        static void Main() {
            GradientLog.OutputWriter = Console.Out;
            GradientEngine.UseEnvironmentFromVariable();

            v1.disable_eager_execution();

            Tensor a = tf.constant(5.0, name: "a");
            Tensor b = tf.constant(10.0, name: "b");

            Tensor sum = tf.add(a, b, name: "sum");
            Tensor div = tf.divide(a, b, name: "div");

            Tensor x = tf.constant(0b101, name: "B101");
            Tensor y = tf.constant(0b011, name: "B011");

            Tensor xor = tf.bitwise.bitwise_xor(x, y);
            Tensor bitcount = gen_bitwise_ops.population_count(xor);

            dynamic config = config_pb2.ConfigProto.CreateInstance();
            // unless this is set, tensorflow-gpu consumes all of GPU memory
            // don't set it if you don't want you training to crash due to random OOM in the middle
            config.gpu_options.allow_growth = true;

            Session session = Session.NewDyn(config: config);
            using var _ = session.StartUsing();

            var writer = new FileWriter(".", session.graph);
            using var __ = writer.StartUsing();

            Console.WriteLine($"a = {session.run(a)}");
            Console.WriteLine($"b = {session.run(b)}");
            Console.WriteLine($"a + b = {session.run(sum)}");
            Console.WriteLine($"a / b = {session.run(div)}");
            Console.WriteLine();

            string xorBinary = Convert.ToString(session.run(xor), toBase: 2).PadLeft(3, '0');
            Console.WriteLine($"101 ^ 011 = {xorBinary} with popcount: {session.run(bitcount)}");
        }
    }
}
