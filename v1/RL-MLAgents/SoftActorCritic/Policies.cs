// ported from https://github.com/openai/spinningup/blob/0cba2886047b7de82c2cad4321df5875db644d61/spinup/algos/tf1/sac/core.py#L29
namespace LostTech.Gradient.Samples.SoftActorCritic {
    using System;
    using LostTech.Gradient;
    using tensorflow;
    using tensorflow.compat.v1;
    using tensorflow.compat.v1.layers;
    using static Tools;

    static class Policies {
        const float LogStdMax = 2;
        const float LogStdMin = -20;

        /// <summary>
        /// Creates a dense neural network, that learns a gaussian distribution over desired actions.
        /// <para>See also: https://www.reddit.com/r/MachineLearning/comments/7fgzfl/d_what_is_gaussian_mlp_policy/ </para>
        /// </summary>
        /// <param name="input">Tensor, representing agent observations</param>
        /// <param name="actionDimensions">Number of action dimentions (e.g. how many degrees
        /// of joystick control the network has)</param>
        /// <param name="hiddenSizes">In a dense layer, sizes of inner layers</param>
        /// <param name="innerActivation">Activation function to use for inner layers of
        /// policy and reward estimation networks. Typically a variant of ReLU (circa 2019).</param>
        /// <param name="outputActivation">Optional extra activation function to use for the output
        /// of policy network. Not needed (e.g. <c>null</c>) circa 2019.</param>
        public static Policy GaussianPolicyNetwork(Tensor input, int actionDimensions, int[] hiddenSizes,
                                                   PythonFunctionContainer innerActivation,
                                                   PythonFunctionContainer? outputActivation) {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (hiddenSizes is null) throw new ArgumentNullException(nameof(hiddenSizes));
            if (innerActivation is null) throw new ArgumentNullException(nameof(innerActivation));

            var network = MultiLayerPreceptron(input, hiddenSizes, innerActivation, innerActivation);
            Tensor mu = layers.dense_dyn(network, units: actionDimensions, activation: outputActivation, name: "mu");

            Tensor logStd;
            using (new variable_scope("logStd").StartUsing()) {
                logStd = layers.dense_dyn(network, units: actionDimensions, activation: tf.tanh_fn);
                logStd = tf.clip_by_value(logStd, clip_value_min: LogStdMin, clip_value_max: LogStdMax);
            }

            Tensor std = tf.exp(logStd, name: "std");
            Tensor pi;
            using(new variable_scope("pi").StartUsing())
                pi = tf.add(mu, tf.random.normal_dyn(tf.shape_dyn(mu)) * std, name: "pi");
            var logpPi = GaussianLikelihood(input: pi, mu: mu, logStd: logStd, name: "logpPi");
            return new Policy(mu: mu, pi: pi, logProbPi: logpPi);
        }

        /// <summary>
        /// Limits the outputs of a policy to avoid overflows in gradients
        /// </summary>
        public static Policy ApplySquashing(Policy policy) {
            var logpPi = tf.subtract(
                policy.logProbPi,
                tf.reduce_sum(
                    (MathF.Log(2) - policy.pi - tf.nn.softplus(-2 * policy.pi))
                    * 2,
                    axis: 1),
                name: "logpPi"
            );

            var mu = tf.tanh(policy.mu, name: "mu");
            var pi = tf.tanh(policy.pi, name: "pi");
            return new Policy(mu: mu, pi: pi, logProbPi: logpPi);
        }
    }
}
