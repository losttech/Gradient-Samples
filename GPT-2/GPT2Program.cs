namespace Gradient.Samples.GPT2
{
    using System;
    using ManyConsole.CommandLineUtils;

    static class Gpt2Program
    {
        static int Main(string[] args)
        {
            Console.Title = "GPT-2";
            GradientSetup.OptInToUsageDataCollection();
            GradientSetup.UseEnvironmentFromVariable();
            // force Gradient initialization
            tensorflow.tf.no_op();

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(Gpt2Program)),
                args, Console.Out);
        }
    }
}
