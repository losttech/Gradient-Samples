namespace LostTech.Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using LostTech.Gradient;
    using mlagents_envs.base_env;
    using numpy;
    /// <summary>
    /// A simple environment, where agents simply have to repeat observations.
    /// <para>(Do what Archimonde does anyone?)</para>
    /// </summary>
    class RepeatObservationEnvironment : IEnvironment {
        float previousObservation;
        float observation;
        float[] action;
        readonly Random random = new Random();
        public RepeatObservationEnvironment(int agents) {
            this.action = new float[agents];
        }
        public int AgentCount => this.action.Length;
        public (DecisionSteps, TerminalSteps?) GetStepResult(string? agentGroupName) {
            if (agentGroupName != null) throw new KeyNotFoundException();
            return (new DecisionSteps(
                obs: new[] { np.ones(new int[] { this.AgentCount, 1 }, dtype: dtype.GetClass<float>()) * this.observation},
                reward: ((ndarray.FromList(this.action) - this.previousObservation).__abs__() - 2).AsArray<float>(),
                agent_id: Enumerable.Range(0, this.AgentCount).ToArray().ToNumPyArray(),
                action_mask: null), null);
        }
        public void Reset() {
            this.observation = this.previousObservation = 0;
            Array.Fill(this.action, 0);
        }
        public void SetActions(string? agentGroupName, ndarray actions) {
            if (agentGroupName != null) throw new KeyNotFoundException();
            for (int agentN = 0; agentN < this.action.Length; agentN++)
                this.action[agentN] = (float32)actions[agentN, 0];
        }
        public void Step() {
            this.previousObservation = this.observation;
            this.observation = (float)this.random.NextDouble() * 2 - 1;
        }

        public static void SanityCheck() {
            // sanity check
            var env = new RepeatObservationEnvironment(agents: 3);
            env.Reset();
            env.Step();
            for(int episode = 0; episode < 100; episode++) {
                var observation = (ndarray)env.GetStepResult(null).Item1.obs[0];
                env.SetActions(null, observation);
                env.Step();
                var step = env.GetStepResult(null);
                var success = step.Item1.reward >= 1.99f;
                bool allPass = success.all();
                Trace.Assert(allPass);
            }
        }
    }
}
