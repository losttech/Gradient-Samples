namespace LostTech.Gradient.Samples {
    using System;
    using System.Collections.Generic;
    using LostTech.Gradient.BuiltIns;

    static class PythonListExtensions {
        public static PythonList<T> ToPyList<T>(this IEnumerable<T> enumerable) {
            if (enumerable is null) throw new ArgumentNullException(nameof(enumerable));
            var result = new PythonList<T>();
            foreach (var item in enumerable)
                result.Add(item);
            return result;
        }
    }
}
