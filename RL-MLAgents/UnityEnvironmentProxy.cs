namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using mlagents_envs.base_env;
    using mlagents_envs.environment;
    using numpy;

    /// <summary>
    /// Implementation of <see cref="IEnvironment"/> for <see cref="UnityEnvironment"/>
    /// </summary>
    class UnityEnvironmentProxy: IEnvironment {
        readonly UnityEnvironment unityEnvironment;
        public UnityEnvironmentProxy(UnityEnvironment unityEnvironment) {
            this.unityEnvironment = unityEnvironment ?? throw new ArgumentNullException(nameof(unityEnvironment));
        }

        public AgentGroupSpec GetAgentGroup(string? name) => this.unityEnvironment.get_agent_group_spec(name);
        public BatchedStepResult GetStepResult(string? agentGroupName) => this.unityEnvironment.get_step_result(agentGroupName);
        public void Reset() => this.unityEnvironment.reset();
        public void SetActions(string? agentGroupName, ndarray actions) => this.unityEnvironment.set_actions(agentGroupName, actions);
        public void Step() => this.unityEnvironment.step();
    }
}
