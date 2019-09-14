namespace Gradient.Samples {
    using System;
    using tensorflow;
    static class ResNetSampleProgram {
        static void Main() {
            GradientLog.OutputWriter = Console.Out;
            GradientSetup.UseEnvironmentFromVariable();

            var block = new ResNetBlock(kernelSize: 1, filters: new[] { 1,2,3 });
            var applicationResult = block.call(tf.zeros(new TensorShape(1, 2, 3, 3)));
            Console.WriteLine(applicationResult);

            foreach(dynamic variable in block.trainable_variables) {
                Console.WriteLine(variable.name);
            }
        }
    }
}
