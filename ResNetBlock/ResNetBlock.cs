namespace Gradient.Samples {
    using System.Collections.Generic;
    using System.Linq;
    using Gradient.BuiltIns;
    using Gradient.ManualWrappers;
    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;

    class ResNetBlock: Model {
        const int PartCount = 3;
        readonly IList<Conv2D> convs = new List<Conv2D>();
        readonly IList<BatchNormalization> batchNorms = new List<BatchNormalization>();
        readonly int outputChannels;
        public ResNetBlock(int kernelSize, int[] filters) {
            for (int part = 0; part < PartCount; part++) {
                this.convs.Add(this.Track(part == 1
                    ? Conv2D.NewDyn(filters[part], kernel_size: kernelSize, padding: "same")
                    : Conv2D.NewDyn(filters[part], kernel_size: (1, 1))));
                this.batchNorms.Add(this.Track(new BatchNormalization()));
            }
            this.outputChannels = filters.Last();
        }

        object CallImpl(IGraphNodeBase inputs, dynamic training) {
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

            result = (Tensor)result + inputs;

            return tf.nn.relu(result);
        }

        public override TensorShape compute_output_shape(TensorShape input_shape) {
            if (input_shape.ndims == 4) {
                var outputShape = input_shape.as_list();
                outputShape[3] = this.outputChannels;
                return new TensorShape(outputShape);
            }

            return input_shape;
        }

        public override dynamic call(IEnumerable<IGraphNodeBase> inputs, ImplicitContainer<IGraphNodeBase> training, IGraphNodeBase mask) {
            return this.CallImpl(inputs.Single(), training);
        }
        public override object call(IEnumerable<IGraphNodeBase> inputs, bool training, IGraphNodeBase mask = null) {
            return this.CallImpl(inputs.Single(), training);
        }

        public override dynamic call(IGraphNodeBase inputs, ImplicitContainer<IGraphNodeBase> training = null, IEnumerable<IGraphNodeBase> mask = null) {
            return this.CallImpl(inputs, training?.Value);
        }
    }
}
