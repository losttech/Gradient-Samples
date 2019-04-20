namespace Gradient.Samples.GPT2 {
    using System.Collections.Generic;
    using numpy;

    public interface IGpt2Decoder<out T>
    {
        T Decode(ndarray tokens);
    }

    public interface IGpt2Encoder<in T>
    {
        List<string> Encode(T value);
    }
}
