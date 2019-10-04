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
        readonly PythonList<Conv2D> convs = new PythonList<Conv2D>();
        readonly PythonList<BatchNormalization> batchNorms = new PythonList<BatchNormalization>();
        readonly PythonFunctionContainer activation;
        readonly int outputChannels;
        public ResNetBlock(int kernelSize, int[] filters, PythonFunctionContainer activation = null) {
            this.activation = activation ?? tf.keras.activations.relu_fn;
            for (int part = 0; part < PartCount; part++) {
                this.convs.Add(this.Track(part == 1
                    ? Conv2D.NewDyn(filters: filters[part], kernel_size: kernelSize, padding: "same")
                    : Conv2D.NewDyn(filters[part], kernel_size: (1, 1))));
                this.batchNorms.Add(this.Track(new BatchNormalization()));
            }

            this.outputChannels = filters[PartCount - 1];
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

            for (int part = 0; part < PartCount; part++) {
                result = this.convs[part].apply(result);
                result = this.batchNorms[part].apply(result, kwargs: batchNormExtraArgs);
                if (part + 1 != PartCount)
                    result = ((dynamic)this.activation)(result);
            }

            result = (Tensor)result + inputs;

            return ((dynamic)this.activation)(result);
        }

        public override dynamic compute_output_shape(TensorShape input_shape) {
            if (input_shape.ndims == 4) {
                var outputShape = input_shape.as_list();
                outputShape[3] = this.outputChannels;
                return new TensorShape(outputShape);
            }

            return input_shape;
        }
    }
}
