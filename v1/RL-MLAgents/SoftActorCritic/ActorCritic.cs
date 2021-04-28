// ported from https://github.com/openai/spinningup/blob/0cba2886047b7de82c2cad4321df5875db644d61/spinup/algos/tf1/sac/core.py#L64
namespace LostTech.Gradient.Samples.SoftActorCritic {
    using System;
    using System.Linq;
    using LostTech.Gradient;
    using tensorflow;
    using tensorflow.compat.v1;
    using PyFunc = Gradient.PythonFunctionContainer;
    using static Tools;

    /// <summary>
    /// Encapsulates neural networks, that represent Soft Actor-Critic-based agent
    /// </summary>
    class ActorCritic {
        /// <summary>
        /// The network(s), that decide what agent is going to do in the environment
        /// </summary>
        public readonly Policy policy;
        /// <summary>
        /// The network(s), that estimate future agent rewards.
        /// </summary>
        /// <seealso cref="Q2"/>
        public readonly Tensor Q1;
        /// <summary>
        /// The network(s), that estimate future agent rewards.
        /// </summary>
        /// <seealso cref="Q1"/>
        public readonly Tensor Q2;

        /// <summary>
        /// Create actor-critic networks
        /// </summary>
        /// <param name="input">Represent agent observations</param>
        /// <param name="action">When training on past experience, this is the action agent tried</param>
        /// <param name="hiddenSizes">Agent policy network here is a simple dense network.
        /// <para>This parameter controls sizes of inner layers.</para></param>
        /// <param name="innerActivation">Activation function to use for inner layers of
        /// policy and reward estimation networks. Typically a variant of ReLU (circa 2019).</param>
        /// <param name="outputActivation">Optional extra activation function to use for the output
        /// of policy network. Not needed (e.g. <c>null</c>) circa 2019.</param>
        /// <param name="policyFactory">A factory function, that creates policy network.
        /// Only one option is provided in the initial sample: <see cref="Policies.GaussianPolicyNetwork"/></param>
        /// <param name="actionLimit">The absolute limit on magnitude of action in all dimensions.</param>
        public ActorCritic(Tensor input, Tensor action, int[] hiddenSizes,
                           PyFunc innerActivation, PyFunc? outputActivation,
                           Func<Tensor, int, int[], PyFunc, PyFunc?, Policy> policyFactory,
                           float actionLimit) {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (action is null) throw new ArgumentNullException(nameof(action));
            if (hiddenSizes is null) throw new ArgumentNullException(nameof(hiddenSizes));
            if (innerActivation is null) throw new ArgumentNullException(nameof(innerActivation));
            if (policyFactory is null) throw new ArgumentNullException(nameof(policyFactory));

            using (new variable_scope("pi").StartUsing()) {
                using(new variable_scope("policy").StartUsing())
                    this.policy = policyFactory(input, action.shape.as_list()[^1].Value, hiddenSizes, innerActivation, outputActivation);
                using(new variable_scope("squashing").StartUsing())
                    this.policy = Policies.ApplySquashing(this.policy);
            }

            this.policy.mu *= actionLimit;
            this.policy.pi *= actionLimit;

            Tensor MakeV(Tensor input)
                => tf.squeeze(MultiLayerPreceptron(input, hiddenSizes.Append(1).ToArray(),
                                                   innerActivation: innerActivation,
                                                   outputActivation: null),
                              axis: 1);

            using (new variable_scope("q1").StartUsing())
                this.Q1 = MakeV(tf.concat(new[] { input, action }, axis: -1));

            using (new variable_scope("q2").StartUsing())
                this.Q2 = MakeV(tf.concat(new[] { input, action }, axis: -1));
        }

        public delegate ActorCritic Factory(
                Tensor input, Tensor action, int[] hiddenSizes,
                PyFunc innerActivation, PyFunc? outputActivation,
                Func<Tensor, int, int[], PyFunc, PyFunc?, Policy> policyFactory,
                float actionLimit);
    }
}
