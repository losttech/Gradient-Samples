namespace BasicMath {
    using System;
    using Python.Runtime;
    using Gradient;
    using tensorflow;
    using tensorflow.summary;

    class BasicMathProgram {
        static void Main(string[] args) {
            var a = new dynamic[] { tf.constant(5.0, name: "a") };
            var b = new dynamic[] { tf.constant(10.0, name: "b") };

            var sum = new dynamic[] { tf.add(a, b, name: "sum") };
            var div = new dynamic[] { tf.div(a, b, name: "div") };

            new Session().UseSelf(session => {
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
