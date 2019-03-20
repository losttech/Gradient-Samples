namespace Gradient.Samples.GPT2
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using MoreLinq;
    using Newtonsoft.Json;
    using numpy;
    using Python.Runtime;
    using static System.Linq.Enumerable;
    public class Gpt2Encoder
    {
        public const string EndOfTextPseudoToken = "<|endoftext|>";

        readonly string errors;
        private readonly IDictionary<string, string> encoder;
        private readonly Dictionary<string, string> decoder;
        readonly Dictionary<byte, char> byteEncoder;
        readonly Dictionary<char, byte> byteDecoder;
        readonly Dictionary<(string,string), float> bpeRanks;

        static readonly dynamic noop = tensorflow.tf.no_op(); // ensure Gradient is initialized
        static readonly dynamic regex = Py.Import("regex");
        static readonly dynamic pattern = regex.compile(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+");
        static readonly Dictionary<byte, char> BytesToUnicode = ComputeBytesToUnicode();

        public string EncodedEndOfText => this.encoder[EndOfTextPseudoToken];

        /// <summary>
        /// <para>     Returns list of utf-8 byte and a corresponding list of unicode strings.
        /// The reversible bpe codes work on unicode strings.
        /// This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        /// When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        /// </para>
        /// <para>
        /// This is a signficant percentage of your normal, say, 32K bpe vocab.
        /// To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        /// </para>
        /// And avoids mapping to whitespace/control characters the bpe code barfs on.
        /// </summary>
        static Dictionary<byte, char> ComputeBytesToUnicode()
        {
            var bs = Range('!', '~' - '!' + 1)
                .Concat(Range('¡', '¬' -'¡' + 1))
                .Concat(Range('®', 'ÿ' - '®' + 1))
                .ToList();
            var cs = bs.ToList();
            int n = 0;
            foreach(int b in Range(0, 256))
            {
                if (bs.Contains(b))
                    continue;

                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
            return bs.EquiZip(cs, (b, c) => ValueTuple.Create(checked((byte)b), checked((char)c))).ToDictionary();
        }
        /// <summary>
        /// Return set of symbol pairs in a word.
        /// </summary>
        /// <returns>Word is represented as tuple of symbols (symbols being variable-length strings).</returns>
        static ISet<(string, string)> GetPairs(string[] word)
        {
            var result = new SortedSet<(string, string)>();
            string prev = word[0];
            foreach(string symbol in word.Skip(1))
            {
                result.Add((prev, symbol));
                prev = symbol;
            }
            return result;
        }

        public Gpt2Encoder(
            IDictionary<string, string> encoder,
            IEnumerable<(string, string)> bpeMerges,
            string errors = "replace")
        {
            this.encoder = encoder;
            this.decoder = encoder.ToDictionary(kv => kv.Value, kv => kv.Key);
            this.errors = errors;
            this.byteEncoder = BytesToUnicode;
            this.byteDecoder = this.byteEncoder.ToDictionary(kv => kv.Value, kv => kv.Key);
            this.bpeRanks = bpeMerges.Select((merge, index) => (merge, (float)index)).ToDictionary();
        }

        readonly Dictionary<string, string> cache = new Dictionary<string, string>();
        string BPE(string token)
        {
            if (this.cache.TryGetValue(token, out var result))
                return result;
            string[] word = token.Select(c => c.ToString()).ToArray();
            var pairs = GetPairs(word);
            if (pairs.Count == 0)
                return token;

            while (true)
            {
                var bigram = pairs.MinBy(pair => this.bpeRanks.GetValueOrDefault(pair, float.PositiveInfinity)).First();
                if (!this.bpeRanks.ContainsKey(bigram))
                    break;

                var (first, second) = bigram;
                var newWord = new List<string>();
                int i = 0;
                while(i < word.Length)
                {
                    int j = Array.IndexOf(word, first, startIndex: i);
                    if (j < 0)
                    {
                        newWord.AddRange(word.Skip(i));
                        break;
                    }

                    newWord.AddRange(word.Skip(i).Take(j - i));
                    i = j;

                    if (word[i] == first && i < word.Length - 1 && word[i + 1] == second)
                    {
                        newWord.Add(first + second);
                        i += 2;
                    } else
                    {
                        newWord.Add(word[i]);
                        i++;
                    }
                }

                word = newWord.ToArray();
                if (word.Length == 1)
                    break;

                pairs = GetPairs(word);
            }

            result = string.Join(" ", word);
            this.cache[token] = result;
            return result;
        }

        public List<string> Encode(string text)
        {
            var bpeTokens = new List<string>();
            foreach(string token in regex.findall(pattern, text))
            {
                string encoded = new string(Encoding.UTF8.GetBytes(token)
                    .Select(@byte => this.byteEncoder[@byte]).ToArray());
                string bpe = this.BPE(encoded);
                foreach (var bpeToken in bpe.Split(' '))
                    bpeTokens.Add(this.encoder[bpeToken]);
            }
            return bpeTokens;
        }

        public string Decode(ndarray tokens)
        {
            string[] tokenStrings = tokens.Cast<object>().Select(t => t.ToString()).ToArray();
            byte[] bytes = tokenStrings.SelectMany(token => this.decoder[token].Select(@char => this.byteDecoder[@char]))
                .ToArray();
            // TODO: error mode!
            return Encoding.UTF8.GetString(bytes);
        }

        public static Gpt2Encoder LoadEncoder(string modelName)
        {
            var encoder = JsonConvert.DeserializeObject<Dictionary<string, string>>(
                File.ReadAllText(Path.Combine("models", modelName, "encoder.json"), Encoding.UTF8));
            string bpeData = File.ReadAllText(Path.Combine("models", modelName, "vocab.bpe"), Encoding.UTF8);
            var bpeMerges = Enumerable.SkipLast(bpeData.Split('\n'), 1).Skip(1)
                .Select(merge => (merge.Split(' ')[0], merge.Split(' ')[1]));
            return new Gpt2Encoder(encoder, bpeMerges);
        }
    }
}
