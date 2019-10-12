namespace Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    static class DataTools {
        public static void Shuffle<T>(this Random rng, T[] array) {
            int n = array.Length;
            while (n > 1) {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }

        public static List<T> Split<T>(ICollection<T> collection, float ratio, out List<T> rest) {
            if (ratio >= 1 || ratio <= 0) throw new ArgumentOutOfRangeException(nameof(ratio));
            int resultLength = (int)(collection.Count * ratio);
            if (resultLength == 0 || resultLength == collection.Count)
                throw new ArgumentException();

            rest = collection.Skip(resultLength).ToList();
            return collection.Take(resultLength).ToList();
        }

        public static T Next<T>(this Random random, IReadOnlyList<T> @from)
            => @from[random.Next(@from.Count)];
        public static T Next<T>(this Random random, IReadOnlyList<T> @from, out int index) {
            index = random.Next(@from.Count);
            return @from[index];
        }
    }
}
