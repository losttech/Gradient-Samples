namespace LostTech.Gradient.Samples.SoftActorCritic {
    using System;
    using tensorflow;

    struct Policy {
        public Tensor mu;
        public Tensor pi;
        public readonly Tensor logProbPi;

        public Policy(Tensor mu, Tensor pi, Tensor logProbPi) {
            this.mu = mu ?? throw new ArgumentNullException(nameof(mu));
            this.pi = pi ?? throw new ArgumentNullException(nameof(pi));
            this.logProbPi = logProbPi ?? throw new ArgumentNullException(nameof(logProbPi));
        }
    }
}
