namespace LostTech.Gradient.Samples.GPT2
{
    using System;
    using LostTech.TensorFlow;
    using ManyConsole.CommandLineUtils;
    using tensorflow.compat.v1;

    static class Gpt2Program
    {
        static int Main(string[] args)
        {
            Console.Title = "GPT-2";
            GradientEngine.UseEnvironmentFromVariable();
            TensorFlowSetup.Instance.EnsureInitialized();

            v1.disable_eager_execution();

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(Gpt2Program)),
                args, Console.Out);
        }
    }
}
