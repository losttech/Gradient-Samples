namespace LostTech.Gradient.Samples.GPT2
{
    using System;
    using LostTech.TensorFlow;
    using ManyConsole.CommandLineUtils;

    static class Gpt2Program
    {
        static int Main(string[] args)
        {
            Console.Title = "GPT-2";
            GradientEngine.UseEnvironmentFromVariable();
            TensorFlowSetup.Instance.EnsureInitialized();

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(Gpt2Program)),
                args, Console.Out);
        }
    }
}
