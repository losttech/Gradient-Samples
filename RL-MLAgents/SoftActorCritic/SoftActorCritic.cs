// ported from https://github.com/openai/spinningup/blob/0cba2886047b7de82c2cad4321df5875db644d61/spinup/algos/tf1/sac/sac.py#L42
namespace Gradient.Samples.SoftActorCritic {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Gradient;
    using numpy;
    using tensorflow;
    using tensorflow.train;
    using static Tools;
    using static System.Linq.Enumerable;
    using CM = Gradient.ContextManagerExtensions;

    static class SoftActorCritic {
        /// <summary>
        /// Creates a Soft Actor-Critic to play the environment and learn
        /// </summary>
        /// <param name="env">Environment to play</param>
        /// <param name="agentGroup">Name of the agent group to control</param>
        /// <param name="actorCriticFactory">A factory, that can create an instance of Soft Actor-Critic</param>
        /// <param name="observationDimensions">Number of dimensions in observation (which assumed
        /// to be an n-dimensional vector)</param>
        /// <param name="actionDimensions">Number of degrees of freedom for agent actions (assumed
        /// to be an m-dimensional vector)</param>
        /// <param name="actionLimit">The absolute limit on magnitude of action in all dimensions.</param>
        /// <param name="actionSampler">Base policy, that just generates random actions.
        /// Used to collect initial observations.</param>
        /// <param name="seed">Setting this number should make experiments repeatable</param>
        /// <param name="stepsPerEpoch">total train steps ==
        /// <paramref name="stepsPerEpoch"/> * <paramref name="epochs"/></param>
        /// <param name="epochs">total train steps ==
        /// <paramref name="stepsPerEpoch"/> * <paramref name="epochs"/></param>
        /// <param name="hiddenSizes">Agent policy network here is a simple dense network.
        /// <para>This parameter controls sizes of inner layers.</para></param>
        /// <param name="batchSize">When training from history, how many experiences to sample
        /// for a single train operation</param>
        /// <param name="startSteps">The number of steps to run random policy for to collect
        /// the initial experience</param>
        /// <param name="replaySize">Maximum number of past experience records to keep in memory.</param>
        /// <param name="updateAfter">Don't start learning until at least this number of steps/ticks
        /// has been observed</param>
        /// <param name="updateEvery">Number of steps/ticks in the environment between
        /// "learning sessions". The agent does not learn after each tick. Instead, it collects
        /// its experiences, then every <paramref name="updateEvery"/> steps/ticks it picks
        /// randomly some past experience from memory, and learns on them.</param>
        /// <param name="gamma"></param>
        /// <param name="polyak"></param>
        /// <param name="learningRate">How impactful new experiences should be.
        /// If set too low, agent will learn very slowly.
        /// If set too high, agent will panically change behavior according to recently picked experience.</param>
        /// <param name="alpha">Affects how random agent's actions will be.</param>
        /// <param name="feedFrames">When training an agent, show it this many frames of observations.
        /// Might be useful to increase this for complex dynamics.</param>
        /// <param name="maxEpisodeLength">Currently unused.</param>
        /// <param name="saveFrequency">Currently unused. Intended to indicate how often to save
        /// the training progress to disk</param>
        public static void Run(IEnvironment env,
                                string? agentGroup,
                                ActorCritic.Factory actorCriticFactory,
                                int observationDimensions,
                                int actionDimensions,
                                float actionLimit,
                                Func<ndarray> actionSampler,
                                int seed = 0,
                                int stepsPerEpoch = 4096, int epochs = 128,
                                int[]? hiddenSizes = null,
                                int batchSize = 128,
                                int startSteps = 8*1024,
                                int replaySize = 1024*1024,
                                int updateAfter = 1024, int updateEvery = 64,
                                float gamma = 0.99f, float polyak = 0.995f,
                                float learningRate = 1e-3f,
                                float alpha = 0.2f,
                                int feedFrames = 1,
                                int maxEpisodeLength = 1024,
                                int saveFrequency = 1) {
            hiddenSizes ??= new int[] { 256, 256 };
            tf.set_random_seed(seed);
            numpy.random.seed((uint)seed);

            env.Reset();
            var stepResult = env.GetStepResult(agentGroup);

            var input = Placeholder(observationDimensions*feedFrames);
            var actionVariable = Placeholder(actionDimensions);
            var input2 = Placeholder(observationDimensions*feedFrames);
            var rewardVariable = Placeholder();
            var doneVariable = Placeholder();

            ActorCritic CreateActorCritic(Tensor input, Tensor action) {
                return actorCriticFactory(input, action,
                    hiddenSizes: hiddenSizes,
                    innerActivation: tf.nn.selu_fn,
                    outputActivation: null,
                    policyFactory: Policies.GaussianPolicyNetwork,
                    actionLimit: actionLimit);
            }

            // Main outputs from computation graph
            ActorCritic actorCritic;
            using(new variable_scope("main").StartUsing())
                actorCritic = CreateActorCritic(input, actionVariable);

            ActorCritic piQ;
            ActorCritic acNext;
            using (new variable_scope("main", reuse: true).StartUsing()) {
                // compose q with pi, for pi-learning
                piQ = CreateActorCritic(input, actorCritic.policy.pi);
                // get actions and log probs of actions for next states, for Q-learning
                acNext = CreateActorCritic(input2, actionVariable);
            }

            ActorCritic target;
            using (new variable_scope("target").StartUsing())
                target = CreateActorCritic(input2, acNext.policy.pi);

            var replayBuffer = new ReplayBuffer(
                observationDimensions: observationDimensions*feedFrames,
                actionDimensions: actionDimensions,
                size: replaySize*stepResult.n_agents(),
                batchSize: stepResult.n_agents());

#if DEBUG
            Console.WriteLine("Number of parameters:");
            foreach (var scope in new[] { "main/pi", "main/q1", "main/q2", "main" })
                Console.WriteLine($"  {scope}: {CountVars(scope)}");
            Console.WriteLine();
            foreach (var scope in new[] { "target/pi", "target/q1", "target/q2", "target" })
                Console.WriteLine($"  {scope}: {CountVars(scope)}");
#endif

            var bestQPi = tf.minimum(piQ.Q1, piQ.Q2);
            var bestQTarget = tf.minimum(target.Q1, target.Q2);

            // Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
            var qBackup = tf.stop_gradient(rewardVariable + gamma * (1 - doneVariable)
                * (bestQTarget - alpha * acNext.policy.logProbPi));

            var piLoss = tf.reduce_mean(alpha * actorCritic.policy.logProbPi - bestQPi, name: "piLoss");
            var q1Loss = 0.5f * tf.reduce_mean(tf.square(qBackup - actorCritic.Q1));
            var q2Loss = 0.5f * tf.reduce_mean(tf.square(qBackup - actorCritic.Q2));

            var valueLoss = q1Loss + q2Loss;

            var piOptimizer = new AdamOptimizer(learning_rate: learningRate, name: "piOpt");
            var pyVars = GetVariables("main/pi").ToPyList();
            Operation trainPi = piOptimizer.minimize(piLoss, var_list: pyVars, name: "trainPi");

            // control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order
            var valueOptimizer = new AdamOptimizer(learning_rate: learningRate, name: "valOpt");
            var valueVars = GetVariables("main/q")
#warning .ToArray() here causes crash in .minimize below
                .ToPyList()
            ;
            Operation trainValue;
            using (var _ = CM.StartUsing(tf.control_dependencies(new[] { trainPi })))
                trainValue = valueOptimizer.minimize(valueLoss, var_list: valueVars, name:"trainVal");

            // Polyak averaging for target variables
            Operation targetUpdate;
            using (var _ = CM.StartUsing(tf.control_dependencies(new[] { trainValue })))
                targetUpdate = tf.group(
                    Enumerable.Select(GetVariables("main").Zip(GetVariables("target")), ((Variable main, Variable target)v)
                    => tf.assign(v.target, v.target * (dynamic)polyak + v.main * (dynamic)(1-polyak), name: "targetUpdate")));

            var targetInit = tf.group(
                GetVariables("main").Zip(GetVariables("target"))
                .Select(((Variable main, Variable target) v) => tf.assign(v.target, v.main)));

            var session = new Session();
            session.run(tf.global_variables_initializer());
            session.run(targetInit);

#warning no model saving
            ndarray GetAction(ndarray observation, bool deterministic) {
                var op = deterministic ? actorCritic.policy.mu : actorCritic.policy.pi;
                return session.run(op, feed_dict: new Dictionary<object, object> {
                    [input] = observation,
                });
            }

            var stopwatch = Stopwatch.StartNew();

            var observation = (ndarray<float>)((ndarray)stepResult.obs[0]).repeat(feedFrames, axis: 1);
            ndarray episodeReward = np.zeros(stepResult.n_agents());
            int episodeLength = 0;
            int totalSteps = stepsPerEpoch * epochs;

            var newObservation = (ndarray<float>)np.zeros_like(observation);
            float aiAction = 0;
            float inducedAction = 0;
            Tools.Dispose(stepResult);
            foreach (int stepN in Range(0, totalSteps)) {
#if DEBUG
                if (stepN == startSteps + 1)
                    Console.WriteLine("\nswitched from random actions to learned policy\n");
#endif
                var action = stepN > startSteps
                    ? GetAction(observation, deterministic: stepN % 2 == 0)
                    : actionSampler();
                aiAction += (float)(float32)action.__abs__().sum();

                env.SetActions(agentGroup, action);
                env.Step();
                var step = env.GetStepResult(agentGroup);
                var newFrame = (ndarray<float>)step.obs[0];
                if (feedFrames > 1) {
                    // TODO: simplifying this depends on https://github.com/dotnet/csharplang/issues/3126
                    for (int agent = 0; agent < step.n_agents(); agent++) {
                        for (int observationDim = 0; observationDim < observationDimensions; observationDim++) {
                            for (int frame = 1; frame < feedFrames; frame++)
                                newObservation[agent, (frame - 1) * observationDimensions + observationDim]
                                    = observation[agent, frame * observationDimensions + observationDim];

                            newObservation[agent, (feedFrames - 1) * observationDimensions + observationDim]
                                = newFrame[agent, observationDim];
                        }
                    }
                    Debug.Assert(newObservation[3, 2].__eq__(observation[3, 2 + observationDimensions]));
                    Tools.Dispose(newFrame);
                } else {
                    newObservation = newFrame;
                }

                var done = (ndarray)step.done.astype(PythonClassContainer<float32>.Instance);
                episodeLength++;
                episodeReward += (dynamic)step.reward;

                replayBuffer.Store(new ReplayBuffer.Observation{
                    observation = observation,
                    newObservation = newObservation,
                    action = action,
                    reward = step.reward,
                    done = done,
                });

                Dispose(action);

                np.copyto(observation, source: newObservation);

                if (stepN >= updateAfter && stepN % updateEvery == 0) {
                    Console.WriteLine($"average reward: {episodeReward.mean(0) / episodeLength}");
                    Console.WriteLine($"ai action: {aiAction} induced action: {inducedAction}");
                    Console.WriteLine($"replay buffer: {replayBuffer.Size*100/replayBuffer.Capacity}%");
                    Console.Write("training...");
                    foreach(int trainingStep in Range(0, updateEvery)) {
                        using var batch = replayBuffer.SampleBatch(batchSize);
                        var feedDict = new Dictionary<object, object> {
                            [input] = batch.observation,
                            [input2] = batch.newObservation,
                            [actionVariable] = batch.action,
                            [rewardVariable] = batch.reward,
                            [doneVariable] = batch.done,
                        };
                        object[] stepOps = {
                            piLoss, q1Loss, q2Loss,
                            actorCritic.Q1, actorCritic.Q2, actorCritic.policy.logProbPi,
                            trainPi, trainValue, targetUpdate,
                        };
                        var outs = session.run(stepOps, feed_dict: feedDict);

                        //tf.io.write_graph(session.graph, nameof(actorCritic), "actorCritic.pbtxt");
                        //Console.Error.WriteLine("written graph");
                        //Environment.Exit(-1);

                        if (trainingStep + 1 == episodeLength)
                            Console.WriteLine($"loss: q1: {outs[1]}; q2: {outs[2]}; logp_pi: {outs[0]}");
                        Dispose(outs);
                    }

                    aiAction = inducedAction = 0;
                    env.Reset();
                    step = env.GetStepResult(agentGroup);
                    observation = (ndarray<float>)((ndarray)step.obs[0]).repeat(feedFrames, axis: 1);
                    episodeReward.fill_dyn(0);
                    episodeLength = 0;
                    Console.WriteLine("\n");
                }

                if (stepN > 0 && stepN % stepsPerEpoch == 0) {
                    int epoch = stepN / stepsPerEpoch;

                    if (epoch % saveFrequency == 0 || epoch == epochs - 1) {
#warning Save model!
                    }
                }
            }
        }
    }
}
