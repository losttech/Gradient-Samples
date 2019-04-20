namespace Gradient.Samples.GPT2 {
    using System.Collections.Generic;
    using numpy;

    public interface IGpt2Encoder<T> {
        List<string> Encode(T value);
        T Decode(ndarray tokens);
    }
}
