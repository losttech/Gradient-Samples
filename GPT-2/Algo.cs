namespace Gradient.Samples.GPT2 {
    using System;
    static class Algo {
        public static int BinarySearch(Func<int, bool> predicate, int lo, int hi) {
            if (predicate is null)
                throw new ArgumentNullException(nameof(predicate));
            if (predicate(lo) || !predicate(hi))
                throw new ArgumentException();
            while(hi > lo + 1) {
                int mid = (lo + hi) / 2;
                if (predicate(mid))
                    hi = mid;
                else
                    lo = mid;
            }
            return hi;
        }
    }
}
