namespace Gradient.Samples {
    using System.Collections.Generic;
    using System.Linq;
    using Gradient.ManualWrappers;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;

    class ResNetBlock: Model {
        const int PartCount = 3;
        readonly IList<Conv2D> convs = new List<Conv2D>();
        readonly IList<BatchNormalization> batchNorms = new List<BatchNormalization>();
        public ResNetBlock(int kernelSize, int[] filters) {
            for (int part = 0; part < PartCount; part++) {
                this.convs.Add(this.Track(part == 1
                    ? Conv2D.NewDyn(filters[part], kernel_size: kernelSize, padding: "same")
                    : Conv2D.NewDyn(filters[part], kernel_size: (1, 1))));
                this.batchNorms.Add(this.Track(new BatchNormalization()));
            }
        }

        public override dynamic call(IEnumerable<IGraphNodeBase> inputs, ImplicitContainer<IGraphNodeBase> training, IGraphNodeBase mask) {
            return this.callImpl((Tensor)inputs.Single(), training);
        }
        public override object call(IEnumerable<IGraphNodeBase> inputs, bool training, IGraphNodeBase mask = null) {
            return this.callImpl((Tensor)inputs, training);
        }

        public override dynamic call(IGraphNodeBase inputs, ImplicitContainer<IGraphNodeBase> training = null, IEnumerable<IGraphNodeBase> mask = null) {
            return this.callImpl(inputs, training?.Value);
        }

        object callImpl(IGraphNodeBase inputs, dynamic training) {
            IGraphNodeBase result = inputs;

            var batchNormExtraArgs = new Dictionary<string, object>();
            if (training != null)
                batchNormExtraArgs["training"] = training;

            for (int part = 0; part < PartCount; part++) {
                result = this.convs[part].__call__(result);
                result = this.batchNorms[part].__call__(result, kwargs: batchNormExtraArgs);
                if (part + 1 != PartCount)
                    result = tf.nn.relu(result);
            }

            result += (Tensor)result + inputs;

            return tf.nn.relu(result);
        }
    }
}
