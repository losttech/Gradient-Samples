namespace LostTech.Gradient.Samples.SoftActorCritic {
    using System;
    using LostTech.Gradient;
    using numpy;
    using static System.Linq.Enumerable;

    /// <summary>
    /// Circular buffer, that records agent observations.
    /// </summary>
    class ReplayBuffer {
        Observation buffer;
        int ptr;
        readonly int batchSize;

        readonly Random random = new Random();
        /// <summary>
        /// Creates new <see cref="ReplayBuffer"/>
        /// </summary>
        /// <param name="observationDimensions">Number of dimensions in observations.
        /// Each observation assumed to be a n-element vector.</param>
        /// <param name="actionDimensions">Number of dimensions in actions.
        /// Each action assumed to be a m-element vector.</param>
        /// <param name="size">Maximum number of records in the buffer.
        /// When this number is reached, older records get overwritten</param>
        /// <param name="batchSize">Number of observations per time step (usually is the number of 
        /// agents)</param>
        public ReplayBuffer(int observationDimensions, int actionDimensions, int size, int batchSize) {
            var dtype = PythonClassContainer<float32>.Instance;
            this.buffer = new Observation {
                observation = np.zeros(new int[] { size, observationDimensions}, dtype: dtype),
                newObservation = np.zeros(new int[] { size, observationDimensions }, dtype: dtype),
                action = np.zeros(new int[] { size, actionDimensions }, dtype: dtype),
                reward= np.zeros(size, dtype: dtype),
                done = np.zeros(size, dtype: dtype),
            };
            this.Capacity = size;
            this.batchSize = batchSize;
            if (size % batchSize != 0)
                throw new ArgumentException($"{nameof(size)} must be mutiplicative of {nameof(batchSize)}");
        }
        /// <summary>
        /// Picks random observations from the recorded history.
        /// </summary>
        /// <param name="batchSize">Number of observations to pick</param>
        public Observation SampleBatch(int batchSize) {
            int[] indices = Range(0, batchSize)
                .Select(_ => this.random.Next(maxValue: this.Size))
                .ToArray();
            return new Observation {
                observation = (ndarray)this.buffer.observation[indices],
                newObservation = (ndarray)this.buffer.newObservation[indices],
                action = (ndarray)this.buffer.action[indices],
                reward = (ndarray)this.buffer.reward[indices],
                done = (ndarray)this.buffer.done[indices],
            };
        }
        /// <summary>
        /// Stores new observations to the buffer, overwriting the oldest ones when necessary
        /// </summary>
        public void Store(Observation observation) {
            if (observation.observation.Length != this.batchSize)
                throw new ArgumentException(
                    message: "The first dimension of input must match batchSize",
                    paramName: nameof(observation));

            foreach (int batchElement in Range(0, this.batchSize)) {
                this.buffer.observation[this.ptr+batchElement] = observation.observation[batchElement];
                this.buffer.newObservation[this.ptr + batchElement] = observation.newObservation[batchElement];
                this.buffer.action[this.ptr + batchElement] = observation.action[batchElement];
                this.buffer.reward[this.ptr + batchElement] = observation.reward[batchElement];
                this.buffer.done[this.ptr + batchElement] = observation.done[batchElement];
            }
            this.ptr = (this.ptr + this.batchSize) % this.Capacity;
            this.Size = Math.Min(this.Size + this.batchSize, this.Capacity);
        }
        /// <summary>
        /// Current number of observations in the buffer
        /// </summary>
        public int Size { get; private set; }
        /// <summary>
        /// Buffer capacity for observations
        /// </summary>
        public int Capacity { get; }

        /// <summary>
        /// Represents an observation of one or several ticks
        /// </summary>
        public struct Observation {
            /// <summary>
            /// What agent saw
            /// </summary>
            public ndarray observation;
            /// <summary>
            /// What agent did
            /// </summary>
            public ndarray action;
            /// <summary>
            /// What reward agent got this time step
            /// </summary>
            public ndarray reward;
            /// <summary>
            /// What agent saw after the tick
            /// </summary>
            public ndarray newObservation;
            /// <summary>
            /// Was the game over after this tick
            /// </summary>
            public ndarray done;
        }
    }
}
