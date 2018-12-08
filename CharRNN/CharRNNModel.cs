namespace CharRNN {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Gradient;
    using SharPy.Runtime;
    using tensorflow;
    using tensorflow.contrib.optimizer_v2.adam;
    using tensorflow.contrib.rnn;
    using tensorflow.nn.rnn_cell;

    class CharRNNModel {
        readonly Random random = new Random();
        readonly CharRNNModelParameters parameters;
        readonly Func<int, RNNCell> cellFactory;
        readonly PythonList<dynamic> cells = new PythonList<dynamic>();
        readonly dynamic cell;
        readonly dynamic inputData;
        internal readonly dynamic initialState;
        internal readonly dynamic trainOp;
        readonly dynamic logits;
        readonly dynamic loss;
        internal readonly dynamic cost;
        internal readonly Either<PythonFunctionContainer, IEnumerable<dynamic>>? finalState;
        readonly dynamic probs;
        internal readonly Variable lr;

        public CharRNNModel(CharRNNModelParameters parameters, bool training = true) {
            this.parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            if (!training) {
                this.parameters.BatchSize = 1;
                this.parameters.SeqLength = 1;
            }

            if (!ModelTypeToCellFunction.TryGetValue(parameters.ModelType, out this.cellFactory))
                throw new NotSupportedException(parameters.ModelType.ToString());

            for(int i = 0; i < parameters.LayerCount; i++) {
                dynamic cell = this.cellFactory(parameters.LayerCount);
                if (training && (parameters.KeepOutputProbability < 1 || parameters.KeepInputProbability < 1))
                    cell = new DropoutWrapper(cell,
                        input_keep_prob: parameters.KeepInputProbability,
                        output_keep_prob: parameters.KeepOutputProbability);
                this.cells.Add(cell);
            }
            this.cell = this.cell = new MultiRNNCell(this.cells, state_is_tuple: true);
            var inputData = tf.placeholder(tf.int32, new TensorShape(parameters.BatchSize, parameters.SeqLength));
            var targets = tf.placeholder(tf.int32, new TensorShape(parameters.BatchSize, parameters.SeqLength));
            var initialState = this.cell.zero_state(parameters.BatchSize, parameters.SeqLength);

            dynamic softmax_W = null, softmax_b = null;
            new variable_scope("rnnlm").UseSelf(_ => {
                softmax_W = tf.get_variable("softmax_w", new TensorShape(parameters.RNNSize, parameters.VocabularySize));
                softmax_b = tf.get_variable("softmax_b", new TensorShape(parameters.VocabularySize));
            });

            var embedding = tf.get_variable("embedding", new TensorShape(parameters.VocabularySize, parameters.RNNSize));
            var inputs = tf.nn.embedding_lookup(embedding, inputData);

            // dropout beta testing: double check which one should affect next line
            if (training && parameters.KeepOutputProbability < 1)
                inputs = tf.nn.dropout(inputs, parameters.KeepOutputProbability);

            inputs = tf.split(inputs, parameters.SeqLength, 1);
            inputs = new PythonList<dynamic>(Enumerable.Select(inputs, i => tf.squeeze(i, 1)));

            dynamic Loop(dynamic prev, dynamic _) {
                prev = tf.matmul(prev, softmax_W) + softmax_b;
                var prevSymbol = tf.stop_gradient(tf.argmax(prev, 1));
                return tf.nn.embedding_lookup(embedding, prevSymbol);
            }
            var (outputs, lastState) = tensorflow.contrib.legacy_seq2seq.rnn_decoder.rnn_decoder_(inputs, initialState, this.cell,
                loop_function: training ? null : Loop, scope: "rnnlm");
            var output = tf.reshape(tf.concat(outputs, 1), new TensorShape(-1, parameters.RNNSize));

            this.logits = tf.matmul(output, softmax_W) + softmax_b;
            var probs = tf.nn.softmax(this.logits);
            this.loss = tensorflow.contrib.legacy_seq2seq.sequence_loss_by_example.sequence_loss_by_example_(
                this.logits,
                tf.reshape(targets, new TensorShape(-1)),
                tf.ones(new Dimension(parameters.BatchSize * parameters.SeqLength)));

            dynamic cost = null;
            new name_scope("cost").UseSelf(_ => {
                cost = tf.reduce_sum(this.loss) / parameters.BatchSize / parameters.SeqLength;
            });
            this.cost = cost;
            var finalState = lastState;
            this.lr = new Variable(0.0, trainable: false);
            var tvars = tf.trainable_variables();

            var (grads, _) = tf.clip_by_global_norm(tf.gradients(this.cost, tvars), parameters.GradientClip);
            AdamOptimizer optimizer = null;
            new name_scope("optimizer").UseSelf(_ => optimizer = new AdamOptimizer(this.lr));
            this.trainOp = optimizer.apply_gradients(grads.Zip(tvars, (g,v) => (dynamic)(g,v)));

            tf.summary.histogram("logits", this.logits);
            tf.summary.histogram("loss", this.loss);
            tf.summary.histogram("train_loss", this.cost);
        }

        public string Sample(Session session, dynamic chars, dynamic vocabulary, int num = 200, string prime = "The ", int samplingType = 1) {
            dynamic state = CreateInitialState(session, vocabulary, prime);

            int WeightedPick(IEnumerable<double> weights) {
                var sums = Enumerable.Aggregate(weights, (sum: 0.0, sums: new List<double>()),
                    (acc, value) => { acc.sum += value; acc.sums.Add(acc.sum); return (acc.sum, acc.sums); }).sums.ToArray();
                int index = Array.BinarySearch(sums, this.random.NextDouble() * sums.Last());
                return index < 0 ? ~index : index;
            }

            var ret = prime;
            var chr = prime.Last();
            for (int i = 0; i < num; i++) {
                var x = np.zeroes((1, 1));
                x[0, 0] = vocabulary[chr];
                var feed = new PythonDict<dynamic, dynamic> {
                    [this.inputData] = x,
                    [this.initialState] = state,
                };
                var outputs = session.run(new dynamic[] { this.probs, this.finalState }, feed);
                var probs = outputs[0];
                state = outputs[1];
                var p = probs[0];

                dynamic sample;
                switch (samplingType) {
                case 1:
                case 2 when chr == ' ':
                    sample = WeightedPick(p);
                    break;
                case 0:
                case 2:
                    sample = np.argmax(p);
                    break;
                default:
                    throw new NotSupportedException();
                }

                var pred = chars[sample];
                ret += pred;
                chr = pred;
            }
            return ret;
        }

        private dynamic CreateInitialState(Session session, dynamic vocabulary, string prime) {
            var state = session.run(this.cell.zero_state(1, tf.float32));
            foreach (var chr in prime.Substring(0, prime.Length - 1)) {
                var x = np.zeroes((1, 1));
                x[0, 0] = vocabulary[chr];
                var feed = new PythonDict<dynamic, dynamic> {
                    [this.inputData] = x,
                    [this.initialState] = state,
                };
                state = Enumerable.First(session.run(this.finalState, feed));
            }

            return state;
        }

        static readonly SortedDictionary<ModelType, Func<int, RNNCell>> ModelTypeToCellFunction = new SortedDictionary<ModelType, Func<int, RNNCell>> {
            [ModelType.RNN] = i => new RNNCell(i),
            [ModelType.GRU] = i => new GRUCell(i),
            [ModelType.LSTM] = i => new LSTMCell(i),
            [ModelType.NAS] = i => new NASCell(i),
        };
    }

    class CharRNNModelParameters {
        public ModelType ModelType { get; set; }
        public int BatchSize { get; set; }
        public int SeqLength { get; set; }
        public int LayerCount { get; set; }
        public int RNNSize { get; set; }
        public int VocabularySize { get; set; }

        public double KeepInputProbability { get; set; }
        public double KeepOutputProbability { get; set; }
        public double GradientClip { get; internal set; }
    }

    enum ModelType {
        RNN,
        GRU,
        LSTM,
        NAS,
    }
}
