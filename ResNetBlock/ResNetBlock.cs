namespace Gradient.Samples {
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
            }
        }

        public override object call(object inputs, bool training, object mask = null) {
            dynamic result = inputs;

            for(int part = 0; part < this.convs.Length; part++) {
                result = this.convs[part].call(result);
                result = this.batchNorms[part].call(result);
                if (part + 1 != this.convs.Length)
                    result = tf.nn.relu(result);
            }

            result += inputs;

            return tf.nn.relu(result);
        }
    }
}
