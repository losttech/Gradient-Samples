# TensorFlow C# samples
Samples for [LostTech.TensorFlow](https://losttech.software/gradient.html), TensorFlow binding for .NET

**BasicMath** ([v1](v1)) - creates two constant tensors and performs simple algebraic operations on them

**CharRNN** - the sample was removed from 2.x samples, as it only works with TensorFlow 1.15.
The old version is [still available in 1.15 branch](https://github.com/losttech/Gradient-Samples/tree/v1.15/CharRNN).

**CSharpOrNot** - a mini-ResNet convolutional network, that guesses programming language,
given a rectangular text block from a code file. Has a cross-platform UI demo.
Get pretrained model here: https://github.com/losttech/Gradient-Samples/releases/tag/csharp-or-not%2Fv1

**GPT-2** ([v1](v1)) - latest published English [language model from OpenAI](https://blog.openai.com/better-language-models/)
(smaller version, pretrained). Added fine-tuning from https://github.com/nshepperd/gpt-2.

**FashionMnistClassification** - standard TensorFlow example, that classifies small pictures of clothes.

**ResNetBlock** - same as FashionMnistClassification above, but shows `Model` subclassing
to implement ResNet block.

**RL-MLAgents** - reinforcement learning agent, that learns to play Unity 3D based games
using Soft Actor-Critic algorithm, and Unity ML agents library. More details in
[the blog post](http://ml.blogs.losttech.software/Reinforcement-Learning-With-Unity-ML-Agents/).

**SimpleApproximation** - uses a simple 1 hidden layer neural network to approximate an arbitrary function.

All models **can be modified and trained**.

**LICENSE** - MIT for all sample code, individual samples might have different licenses (clearing that up, see individual sample folders).

# See Also

[**SIREN**](https://github.com/losttech/Siren) - neural representation
for any kind of signal (image, video, audio).

[**YOLOv4**](https://github.com/losttech/YOLOv4) - neural network for object detection.

[**Billion Songs**](https://github.com/losttech/BillionSongs) -
deep learning-powered song lyrics generator in an ASP.NET Core web site.
More details in
[Writing billion songs with C# and Deep Learning](https://habr.com/post/453232/).
