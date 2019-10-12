namespace Gradient.Samples {
    using Avalonia;
    using JetBrains.Annotations;
    using ManyConsole.CommandLineUtils;

    [UsedImplicitly]
    class UICommand: ConsoleCommand {
        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        public override int Run(string[] remainingArguments) {
            CSharpOrNotProgram.BuildAvaloniaApp().Start(AppMain, remainingArguments);
            return 0;
        }

        // Your application's entry point. Here you can initialize your MVVM framework, DI
        // container, etc.
        static void AppMain(Application app, string[] args) {
            app.Run(new CSharpOrNotWindow());
        }

        public UICommand() {
            this.IsCommand("ui");
            this.IsCommand("run");
        }
    }
}
