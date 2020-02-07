namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using Gradient;
    using Gradient.Samples.SoftActorCritic;
    using LostTech.TensorFlow;
    using mlagents_envs.environment;
    using mlagents_envs.side_channel.engine_configuration_channel;
    using numpy;
    using tensorflow;
    using PyFunc = Gradient.PythonFunctionContainer;
    using static System.Linq.Enumerable;

    static class Program {
        static ActorCritic ActorCriticFactory(
                    Tensor input, Tensor action, int[] hiddenSizes,
                    PyFunc innerActivation, PyFunc? outputActivation,
                    Func<Tensor, int, int[], PyFunc, PyFunc?, Policy> policyFactory,
                    float actionLimit)
                    => new ActorCritic(input, action, hiddenSizes,
                                       innerActivation, outputActivation,
                                       policyFactory, actionLimit);

        static void Main(string[] args) {
            // deploy prepackaged TensorFlow runtime next to this assembly
            var deployTensorFlowTo = new DirectoryInfo(Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location),
                "TensorFlow"));
            var tfEnv = PackagedTensorFlow.EnsureDeployed(deployTensorFlowTo);

            GradientEngine.UseEnvironment(tfEnv);
            GradientSetup.EnsureInitialized();
            Console.Title = "ML Agents";

            if (args.Length == 1)
                SetCpuAffinity(limitCoresTo: int.Parse(args[0]));

            // debug environment, that does not require Unity: goal is to simply repeat observation
            // RunRepeat();

            var engineConfigChannel = new EngineConfigurationChannel();
            // connect to running Unity, you'll have to press Play there
            const string? envName = null;
            var env = new UnityEnvironment(base_port: 5004, file_name: envName,
                side_channels: new[] { engineConfigChannel });

            try {
                engineConfigChannel.set_configuration_parameters(time_scale: 0.3);
                env.reset();

                string agentGroup = env.get_agent_groups().Single();
                var spec = env.get_agent_group_spec(agentGroup);

                var stepResult = env.get_step_result(agentGroup);
                Debug.Assert(stepResult.obs.Count == 1);
                (int, int) observationsShape = ((ndarray)stepResult.obs[0]).shape;
                int observationSize = observationsShape.Item2;

                if (!spec.is_action_continuous())
                    throw new NotImplementedException("discrete");

                var random = new Random();
                ndarray RandomActionSampler()
                    // a list of random values between -1.0 and +1.0
                    => (ndarray)ndarray.FromList(Range(0, spec.action_size * stepResult.n_agents())
                        .Select(_ => (float)random.NextDouble() * 2 - 1)
                        .ToList())
                    .reshape(new int[] { stepResult.n_agents(), spec.action_size })
                    .astype(PythonClassContainer<float32>.Instance);

                SoftActorCritic.SoftActorCritic.Run(new UnityEnvironmentProxy(env),
                    agentGroup: agentGroup,
                    actorCriticFactory: ActorCriticFactory,
                    observationDimensions: observationSize,
                    actionDimensions: spec.action_size,
                    actionLimit: 1,
                    feedFrames: 1,
                    maxEpisodeLength: 1024,
                    startSteps: 2048,
                    replaySize: 1024 * 1024 / 8,
                    actionSampler: RandomActionSampler);
            } finally {
                env.close();
            }
        }

        static void RunRepeat() {
            RepeatObservationEnvironment.SanityCheck();

            var random = new Random();
            const int RepeatAgents = 3;
            ndarray RepeatRandomActionSampler()
                => (ndarray)ndarray.FromList(Range(0, RepeatAgents)
                    .Select(_ => (float)random.NextDouble() * 2 - 1)
                    .ToList())
                .reshape(new int[] { RepeatAgents, 1 })
                .astype(PythonClassContainer<float32>.Instance);
            SoftActorCritic.SoftActorCritic.Run(new RepeatObservationEnvironment(RepeatAgents),
                agentGroup: null,
                actorCriticFactory: ActorCriticFactory,
                observationDimensions: 1,
                actionDimensions: 1,
                actionLimit: 1,
                feedFrames: 1,
                hiddenSizes: new int[] { 32 },
                maxEpisodeLength: 256,
                replaySize: 1024 * 1024 / 8,
                learningRate: 2e-4f,
                startSteps: 100,
                actionSampler: RepeatRandomActionSampler);
        }

        static void SetCpuAffinity(int limitCoresTo) {
            var self = Process.GetCurrentProcess();
            self.ProcessorAffinity = new IntPtr((1L << limitCoresTo) - 1);
        }
    }
}
