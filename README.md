# Gradient-Samples
Samples for [Gradient](https://losttech.software/gradient.html), TensorFlow binding for .NET

[Billion Songs](https://github.com/losttech/BillionSongs) a repository with
deep learning-powered song lyrics generator in an ASP.NET Core web site.

**BasicMath**- creates two constant tensors and performs simple algebraic operations on them

**CharRNN** - generates semi-sensical text in the style of input. For example (Shakespeare):
> LOST:
> What is your name beautiful princess?
>
> COMPUTER:
> Gradient your horn of exemity.
>
> GLOUCESTER:
> how one sick up scourish, thither place and thronds the grow for these bowing.
> 
> MARIANA:
> He shall speak not reverbering injurance.

**GPT-2** - latest English [language model from OpenAI](https://blog.openai.com/better-language-models/)
(smaller version, pretrained). Added fine-tuning from https://github.com/nshepperd/gpt-2.

**FashionMnistClassification** - standard TensorFlow example, that classifies small pictures of clothes.

**LinearSVM** - incomplete port due to a bug with tf.flags

**SimpleApproximation** - uses a simple 1 hidden layer neural network to approximate an arbitrary function.

All models **can be modified and trained**.

**LICENSE** - MIT, individual samples might have different licenses (clearing that up, see individual sample folders).
