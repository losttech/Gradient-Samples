// ported from https://github.com/openai/spinningup/blob/0cba2886047b7de82c2cad4321df5875db644d61/spinup/algos/tf1/sac/core.py#L1
namespace LostTech.Gradient.Samples.SoftActorCritic {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using LostTech.Gradient;
    using LostTech.Gradient.BuiltIns;
    using tensorflow;
    using tensorflow.compat.v1;
    using tensorflow.compat.v1.layers;
    using Variable = tensorflow.Variable;

    static class Tools {
        const float Epsilon = 1e-08f;
        static DType defaultType = tf.float32;
        public static Tensor Placeholder(int size)
            => v1.placeholder(defaultType, new TensorShape(null, size));
        public static Tensor Placeholder()
            => v1.placeholder(defaultType, new TensorShape(new int?[] { null }));
        /// <summary>
        /// Creates a dense neural network
        /// </summary>
        public static Tensor MultiLayerPreceptron(Tensor input, int[] hiddenSizes,
                                                  PythonFunctionContainer innerActivation,
                                                  PythonFunctionContainer? outputActivation) {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (hiddenSizes is null) throw new ArgumentNullException(nameof(hiddenSizes));
            if (innerActivation is null) throw new ArgumentNullException(nameof(innerActivation));

            for (int layer = 0; layer < hiddenSizes.Length; layer++) {
                input = layers.dense_dyn(input,
                    units: hiddenSizes[layer],
                    activation: layer == hiddenSizes.Length - 1 ? outputActivation : innerActivation
                );
            }
            return input;
        }
        public static IEnumerable<Variable> GetVariables(string scopeNamePrefix)
            => ((PythonList<Variable>)v1.global_variables()).Where(v => v.name.Contains(scopeNamePrefix));
        public static int CountVars(string scopeNamePrefix) {
            var variables = GetVariables(scopeNamePrefix);
            return variables.Sum(v => v.shape.as_list().Cast<int>().Aggregate((a, b) => a * b));
        }

        public static Tensor GaussianLikelihood(Tensor input, Tensor mu, Tensor logStd, string? name = null) {
            if (input is null) throw new ArgumentNullException(nameof(input));
            if (mu is null) throw new ArgumentNullException(nameof(mu));
            if (logStd is null) throw new ArgumentNullException(nameof(logStd));

            using var _ = new variable_scope("gaussian_likelihood").StartUsing();
            var preSum = (tf.square((input - mu) / (tf.exp(logStd) + Epsilon))
                                 + logStd * 2
                                 + MathF.Log(2 * MathF.PI))
                         * -0.5;
            return tf.reduce_sum(preSum, axis: 1, name: name);
        }

        public static Tensor ClipButPassGradient(Tensor input, float min, float max) {
            using var _ = new variable_scope("clip_val_pass_grad").StartUsing();
            Tensor clippedMax = tf.cast(input > max, tf.float32);
            Tensor clippedMin = tf.cast(input < min, tf.float32);
            return input + tf.stop_gradient(((max - input) * clippedMax)
                                          + ((min - input) * clippedMin));
        }
    }
}
