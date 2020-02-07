namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using mlagents_envs.base_env;
    using numpy;

    /// <summary>
    /// A simple environment interface
    /// </summary>
    interface IEnvironment {
        /// <summary>
        /// Reset the environment to its initial stage
        /// </summary>
        void Reset();
        /// <summary>
        /// Advance time by 1 tick/step in the environment
        /// </summary>
        void Step();
        /// <summary>
        /// Get information about the last tick/step in the environment, as seen by a group of agents
        /// </summary>
        BatchedStepResult GetStepResult(string? agentGroupName);
        /// <summary>
        /// Call before <see cref="Step"/> to set the action(s), that agent(s) in the specified
        /// group will do during the next time step
        /// </summary>
        void SetActions(string? agentGroupName, ndarray actions);
    }
}
