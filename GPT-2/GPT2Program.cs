namespace Gradient.Samples.GPT2
{
    using System;
    using ManyConsole.CommandLineUtils;

    static class Gpt2Program
    {
        static int Main(string[] args)
        {
            GradientSetup.OptInToUsageDataCollection();
            // force Gradient initialization
            tensorflow.tf.no_op();

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(Gpt2Program)),
                args, Console.Out);
        }
    }
}
