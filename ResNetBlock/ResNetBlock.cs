namespace Gradient.Samples {
    using System.Collections.Generic;
    using System.Linq;
    using Gradient.ManualWrappers;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;

    class ResNetBlock: Model {
        const int PartCount = 3;
        readonly Conv2D[] convs = new Conv2D[PartCount];
        readonly BatchNormalization[] batchNorms = new BatchNormalization[PartCount];
        public ResNetBlock(int kernelSize, int[] filters) {
            for (int part = 0; part < this.convs.Length; part++) {
                this.convs[part] = part == 1
                    ? Conv2D.NewDyn(filters[part], kernel_size: kernelSize, padding: "same")
                    : Conv2D.NewDyn(filters[part], kernel_size: (1, 1));
                this.batchNorms[part] = new BatchNormalization();
            }
        }

        public override dynamic call(IEnumerable<IGraphNodeBase> inputs, ImplicitContainer<IGraphNodeBase> training, IGraphNodeBase mask) {
            return this.callImpl((Tensor)inputs.Single(), training);
        }

        public override object call(object inputs, bool training, IGraphNodeBase mask = null) {
            return this.callImpl((Tensor)inputs, training);
        }

        public override dynamic call(object inputs, ImplicitContainer<IGraphNodeBase> training = null, IEnumerable<IGraphNodeBase> mask = null) {
            return this.callImpl((Tensor)inputs, training?.Value);
        }

        object callImpl(IGraphNodeBase inputs, dynamic training) {
            IGraphNodeBase result = inputs;

            var batchNormExtraArgs = new PythonDict<string, object>();
            if (training != null)
                batchNormExtraArgs["training"] = training;

            for (int part = 0; part < this.convs.Length; part++) {
                result = this.convs[part].__call__(result);
                result = this.batchNorms[part].__call__(result, kwargs: batchNormExtraArgs);
                if (part + 1 != this.convs.Length)
                    result = tf.nn.relu(result);
            }

            result += (Tensor)result + inputs;

            return tf.nn.relu(result);
        }
    }
}
