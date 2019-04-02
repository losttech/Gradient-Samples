namespace LinearSVM
{
    using ManyConsole.CommandLineUtils;
    class LinearSvmCommand: ConsoleCommand
    {
        public int BatchSize { get; set; } = 32;
        public int StepCount { get; set; } = 500;
        public bool IsEvaluation { get; set; } = true;
        /// <summary>
        /// Penalty parameter of the error term.
        /// </summary>
        public double C { get; set; } = 0.1;
        /// <summary>
        /// Penalty parameter of the error term.
        /// </summary>
        public double Reg { get; set; } = 1.0;
        /// <summary>
        /// Margin?
        /// </summary>
        public double Delta { get; set; } = 1;
        public double InitialLearningRate { get; set; } = 0.1;

        public LinearSvmCommand()
        {
            this.IsCommand("run");
            this.HasOption("b|batch-size=", "Batch size", (int size) => this.BatchSize = size);
            this.HasOption("c|step-count=", "Step count", (int steps) => this.StepCount = steps);
            this.HasOption("t|train", "Train", _ => this.IsEvaluation = !this.IsEvaluation);
            this.HasOption("C=", "C parameter", (double c) => this.C = c);
            this.HasOption("Reg=", "Reg parameter", (double r) => this.Reg = r);
            this.HasOption("delta=", "Delta", (double d) => this.Delta = d);
            this.HasOption("lr|initial-learning-rate=", "Initial learning rate",
                (double lr) => this.InitialLearningRate = lr);
        }

        public override int Run(string[] remainingArguments)
        {
            return new LinearSvmProgram(this).Run();
        }
    }
}
