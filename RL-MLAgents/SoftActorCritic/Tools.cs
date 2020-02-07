// ported from https://github.com/openai/spinningup/blob/0cba2886047b7de82c2cad4321df5875db644d61/spinup/algos/tf1/sac/core.py#L1
namespace Gradient.Samples.SoftActorCritic {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Gradient;
    using tensorflow;

    static class Tools {
        const float Epsilon = 1e-08f;
        static DType defaultType = tf.float32;
        public static Tensor Placeholder(int size)
            => tf.placeholder(defaultType, new TensorShape(null, size));
        public static Tensor Placeholder()
            => tf.placeholder(defaultType, new TensorShape(new int?[] { null }));
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
                input = tf.layers.dense(input,
                    units: hiddenSizes[layer],
                    activation: layer == hiddenSizes.Length - 1 ? outputActivation : innerActivation
                );
            }
            return input;
        }
        public static IEnumerable<Variable> GetVariables(string scopeNamePrefix)
            => Enumerable.Where(((IEnumerable<Variable>)tf.global_variables()), v => v.name.Contains(scopeNamePrefix));
        public static int CountVars(string scopeNamePrefix) {
            var variables = GetVariables(scopeNamePrefix);
            return variables.Sum(v => ((TensorShape)v.shape).dims.Select(d => d.__long__().Value).Aggregate((a, b) => a * b));
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

        public static void Dispose(params IDisposable[] disposables) {
            foreach (var disposable in disposables)
                disposable.Dispose();
        }

        // Our preview version of Python.NET leaks memory. We should dispose large objects explicitly.
        public static void Dispose(params PythonObjectContainer[] containers) {
            foreach (var container in containers)
                container.PythonObject.Dispose();
        }
    }
}
