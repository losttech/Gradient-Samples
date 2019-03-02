namespace Gradient.Samples.GPT2
{
    using System;
    using System.Runtime.CompilerServices;
    using Python.Runtime;

    static class RangeWorkaround
    {
        static readonly PyObject slice  = PythonEngine.Eval("slice");
        static readonly PyObject none = PythonEngine.Eval("None");
        static PyObject Pythonify(Range range)
        {
            // in C# ranges are inclusive
            // in Python the right end is excluded
            int? end = range.End.IsFromEnd
                ? (range.End.Value == 0 ? (int?)null : ToPythonIndex(range.End) + 1)
                : ToPythonIndex(range.End) + 1;
            return slice.Invoke(
                Pythonify(ToPythonIndex(range.Start)),
                end == null ? none : Pythonify(end.Value));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static PyObject Pythonify(int index) => index.ToPython();
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int ToPythonIndex(Index index) => index.IsFromEnd ? ~index.Value : index.Value;

        public static PyObject All() => Pythonify(Range.All);
        public static PyObject FromStart(Index start) => Pythonify(Range.StartAt(start));
        public static PyObject ToEnd(Index end) => Pythonify(Range.EndAt(end));
    }
}
