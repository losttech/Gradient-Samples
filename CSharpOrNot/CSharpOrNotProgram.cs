namespace LostTech.Gradient.Samples {
    using System;
    using System.Linq;
    using Avalonia;
    using Avalonia.Logging.Serilog;
    using LostTech.Gradient;
    using LostTech.TensorFlow;
    using ManyConsole.CommandLineUtils;
    using tensorflow;
    using tensorflow.core.protobuf.config_pb2;

    static class CSharpOrNotProgram {
        public static int Main(string[] args) {
            TensorFlowSetup.Instance.OptInToUsageDataCollection();
            GradientEngine.UseEnvironmentFromVariable();

            var gpus = tf.config.list_physical_devices("GPU");
            foreach(var gpu in gpus)
                tf.config.experimental.set_memory_growth(gpu, enable: true);

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(CSharpOrNotProgram)),
                args, Console.Out);
        }

        // Avalonia configuration, don't remove; also used by visual designer.
        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .LogToDebug();
    }
}
