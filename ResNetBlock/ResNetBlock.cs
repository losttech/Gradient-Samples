namespace LostTech.Gradient.Samples {
    using System.Collections.Generic;
    using System.Linq;
    using LostTech.Gradient.ManualWrappers;

    using tensorflow;
    using tensorflow.keras;
    using tensorflow.keras.layers;

    public class ResNetBlock: Model {
        const int PartCount = 3;
        readonly List<Conv2D> convs = new List<Conv2D>();
        readonly List<BatchNormalization> batchNorms = new List<BatchNormalization>();
        readonly PythonFunctionContainer activation;
        readonly int outputChannels;
        public ResNetBlock(int kernelSize, int[] filters, PythonFunctionContainer? activation = null) {
            this.activation = activation ?? tf.keras.activations.relu_fn;
            for (int part = 0; part < PartCount; part++) {
                this.convs.Add(this.Track(part == 1
                    ? Conv2D.NewDyn(filters: filters[part], kernel_size: kernelSize, padding: "same")
                    : Conv2D.NewDyn(filters[part], kernel_size: (1, 1))));
                this.batchNorms.Add(this.Track(new BatchNormalization()));
            }

            this.outputChannels = filters[PartCount - 1];
        }

        Tensor CallImpl(IGraphNodeBase inputs, dynamic? training) {
            IGraphNodeBase result = inputs;

            var batchNormExtraArgs = new Dictionary<string, object>();
            if (!(training is null))
                batchNormExtraArgs["training"] = training;

            for (int part = 0; part < PartCount; part++) {
                result = this.convs[part].__call__(result);
                result = this.batchNorms[part].__call__(result, kwargs: batchNormExtraArgs);
                if (part + 1 != PartCount)
                    result = this.activation.Invoke(result)!;
            }

            result = (Tensor)result + inputs;

            return this.activation.Invoke(result)!;
        }

        public override TensorShape compute_output_shape(TensorShape input_shape) {
            if (input_shape.ndims == 4) {
                var outputShape = input_shape.as_list();
                outputShape[3] = this.outputChannels;
                return new TensorShape(outputShape);
            }

            return input_shape;
        }

        public override Tensor call(IEnumerable<IGraphNodeBase> inputs, IGraphNodeBase? training, IGraphNodeBase? mask) {
            return this.CallImpl((Tensor)inputs.Single(), training);
        }

        public override Tensor call(IGraphNodeBase inputs, bool training, IGraphNodeBase? mask = null) {
            return this.CallImpl((Tensor)inputs, training);
        }

        public override Tensor call(IGraphNodeBase inputs, IGraphNodeBase? training = null, IEnumerable<IGraphNodeBase>? mask = null) {
            return this.CallImpl((Tensor)inputs, training);
        }
    }
}
