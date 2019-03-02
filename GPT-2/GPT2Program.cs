namespace Gradient.Samples.GPT2
{
    using System;
    static class Gpt2Program
    {
        static void Main()
        {
            GradientSetup.OptInToUsageDataCollection();
            // force Gradient initialization
            tensorflow.tf.no_op();

            Gpt2Interactive.Run();
        }
    }
}
