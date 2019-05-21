namespace CharRNN {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using CommandLine;
    using Gradient;
    using SharPy.Runtime;
    using numpy;
    using tensorflow;
    using tensorflow.contrib.optimizer_v2.adam;
    using tensorflow.contrib.rnn;
    using tensorflow.nn.rnn_cell;
    using seq2seqState = System.ValueTuple<tensorflow.nn.rnn_cell.LSTMStateTuple, tensorflow.nn.rnn_cell.LSTMStateTuple>;

    class CharRNNModel {
        readonly Random random = new Random();
        readonly CharRNNModelParameters parameters;
        readonly Func<int, RNNCell> cellFactory;
        readonly PythonList<RNNCell> cells = new PythonList<RNNCell>();
        readonly RNNCell rnn;
        internal readonly dynamic inputData;
        internal readonly seq2seqState initialState;
        internal readonly dynamic trainOp;
        readonly Tensor logits;
        readonly Tensor loss;
        internal readonly Tensor cost;
        internal readonly seq2seqState finalState;
        readonly dynamic probs;
        internal readonly dynamic targets;
        internal readonly Variable learningRate;

        public CharRNNModel(CharRNNModelParameters parameters, bool training = true) {
            this.parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            if (!training) {
                this.parameters.BatchSize = 1;
                this.parameters.SeqLength = 1;
            }

            if (!ModelTypeToCellFunction.TryGetValue(parameters.ModelType, out this.cellFactory))
                throw new NotSupportedException(parameters.ModelType.ToString());

            for(int i = 0; i < parameters.LayerCount; i++) {
                RNNCell cell = this.cellFactory(parameters.RNNSize);
                if (training && (parameters.KeepOutputProbability < 1 || parameters.KeepInputProbability < 1))
                    cell = new DropoutWrapper(cell,
                        input_keep_prob: parameters.KeepInputProbability,
                        output_keep_prob: parameters.KeepOutputProbability);
                this.cells.Add(cell);
            }
            this.rnn = new MultiRNNCell(this.cells, state_is_tuple: true);
            this.inputData = tf.placeholder(tf.int32, new TensorShape(parameters.BatchSize, parameters.SeqLength));
            this.targets = tf.placeholder(tf.int32, new TensorShape(parameters.BatchSize, parameters.SeqLength));
            this.initialState = this.rnn.zero_state(parameters.BatchSize, tf.float32);

            Variable softmax_W = null, softmax_b = null;
            new variable_scope("rnnlm").UseSelf(_ => {
                softmax_W = tf.get_variable("softmax_w", new TensorShape(parameters.RNNSize, parameters.VocabularySize));
                softmax_b = tf.get_variable("softmax_b", new TensorShape(parameters.VocabularySize));
            });

            Variable embedding = tf.get_variable("embedding", new TensorShape(parameters.VocabularySize, parameters.RNNSize));
            Tensor input = tf.nn.embedding_lookup(embedding, this.inputData);

            // dropout beta testing: double check which one should affect next line
            if (training && parameters.KeepOutputProbability < 1)
                input = tf.nn.dropout(input, parameters.KeepOutputProbability);

            PythonList<Tensor> inputs = tf.split(input, parameters.SeqLength, axis: 1);
            inputs = inputs.Select(i => (Tensor)tf.squeeze(i, axis: 1)).ToPythonList();

            dynamic Loop(dynamic prev, dynamic _) {
                prev = tf.matmul(prev, softmax_W) + softmax_b;
                var prevSymbol = tf.stop_gradient(tf.argmax(prev, 1));
                return tf.nn.embedding_lookup(embedding, prevSymbol);
            }
            var decoder = tensorflow.contrib.legacy_seq2seq.legacy_seq2seq.rnn_decoder_dyn(
                decoder_inputs: inputs,
                initial_state: this.initialState.Items(),
                cell: this.rnn,
                loop_function: training ? null : PythonFunctionContainer.Of(new Func<dynamic, dynamic, dynamic>(Loop)), scope: "rnnlm");
            var outputs = decoder.Item1;
            var lastState = (seq2seqState)decoder.Item2;
            dynamic contatenatedOutputs = tf.concat(outputs, 1);
            var output = tensorflow.tf.reshape(contatenatedOutputs, new[] { -1, parameters.RNNSize });

            this.logits = tf.matmul(output, softmax_W) + softmax_b;
            this.probs = tf.nn.softmax(new[] { this.logits });
            this.loss = tensorflow.contrib.legacy_seq2seq.legacy_seq2seq.sequence_loss_by_example_dyn(
                logits: new[] { this.logits },
                targets: new[] { tf.reshape(this.targets, new[] { -1 }) },
                weights: new[] { tf.ones(new[] { parameters.BatchSize * parameters.SeqLength }) });

            Tensor cost = null;
            new name_scope("cost").UseSelf(_ => {
                cost = tf.reduce_sum(this.loss) / parameters.BatchSize / parameters.SeqLength;
            });
            this.cost = cost;
            this.finalState = lastState;
            this.learningRate = new Variable(0.0, trainable: false);
            var tvars = tf.trainable_variables();

            IEnumerable<object> grads = tf.clip_by_global_norm(tf.gradients(this.cost, tvars), parameters.GradientClip).Item1;
            AdamOptimizer optimizer = null;
            new name_scope("optimizer").UseSelf(_ => optimizer = new AdamOptimizer(this.learningRate));
            this.trainOp = optimizer.apply_gradients(grads.Zip(tvars, (grad, @var) => (dynamic)(grad, @var)));

            tf.summary.histogram("logits", new[] { this.logits });
            tf.summary.histogram("loss", new[] { this.loss });
            tf.summary.histogram("train_loss", new[] { this.cost });
        }

        public string Sample(Session session, dynamic chars, IReadOnlyDictionary<char, int> vocabulary, int num = 200, string prime = "The ", int samplingType = 1) {
            dynamic state = this.CreateInitialState(session, vocabulary, prime);

            int WeightedPick(IEnumerable<float32> weights) {
                double[] sums = weights.Aggregate((sum: 0.0, sums: new List<double>()),
                    (acc, value) => {
                        acc.sum += (double)value; acc.sums.Add(acc.sum);
                        return (acc.sum, acc.sums);
                    }).sums.ToArray();
                int index = Array.BinarySearch(sums, this.random.NextDouble() * sums.Last());
                return index < 0 ? ~index : index;
            }

            string ret = prime;
            char chr = prime.Last();
            for (int i = 0; i < num; i++) {
                var x = np.zeros(new TensorShape(1, 1));
                x[0, 0] = vocabulary[chr];
                var feed = new PythonDict<dynamic, dynamic> {
                    [this.inputData] = x,
                    [this.initialState] = state,
                };
                var outputs = session.run(new dynamic[] { this.probs, this.finalState }, feed);
                var probs = outputs[0];
                state = outputs[1];
                ndarray computedProbabilities = probs[0];

                dynamic sample;
                switch (samplingType) {
                case 1:
                case 2 when chr == ' ':
                    sample = WeightedPick(computedProbabilities.Cast<ndarray>().SelectMany(s => s.Cast<float32>()));
                    break;
                case 0:
                case 2:
                    sample = computedProbabilities.argmax();
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

        private dynamic CreateInitialState(Session session, IReadOnlyDictionary<char,int> vocabulary, string prime) {
            var state = session.run(this.rnn.zero_state(1, tf.float32));
            foreach (char chr in prime.Substring(0, prime.Length - 1)) {
                var x = np.zeros(new TensorShape(1, 1));
                x[0, 0] = vocabulary[chr];
                var feed = new PythonDict<dynamic, dynamic> {
                    [this.inputData] = x,
                    [this.initialState] = state,
                };
                state = Enumerable.First(session.run(new dynamic[] { this.finalState }, feed));
            }

            return state;
        }

        static readonly SortedDictionary<ModelType, Func<int, RNNCell>> ModelTypeToCellFunction = new SortedDictionary<ModelType, Func<int, RNNCell>> {
            [ModelType.RNN] = i => RNNCell.NewDyn(i),
            [ModelType.GRU] = i => new GRUCell(i),
            [ModelType.LSTM] = i => new LSTMCell(i),
            [ModelType.NAS] = i => new NASCell(i),
        };
    }

    class CharRNNModelParameters {
        [Option("type", Default = ModelType.LSTM)]
        public ModelType ModelType { get; set; }
        [Option("batch", Default = 50)]
        public int BatchSize { get; set; }
        [Option("seq-len", Default = 50, HelpText = "RNN sequence length. Number of timesteps to unroll for.")]
        public int SeqLength { get; set; }
        [Option("layers", Default = 2)]
        public int LayerCount { get; set; }
        [Option("rnn-size", Default = 128, HelpText = "size of RNN hidden state")]
        public int RNNSize { get; set; }
        public int VocabularySize { get; set; }

        [Option("keep-in-prob", Default = 1.0)]
        public double KeepInputProbability { get; set; }
        [Option("keep-out-prob", Default = 1.0)]
        public double KeepOutputProbability { get; set; }
        [Option('c', "gradient-clip", Default = 5)]
        public double GradientClip { get; internal set; }
    }

    public enum ModelType {
        RNN,
        GRU,
        LSTM,
        NAS,
    }
}
