namespace Gradient.Samples.GPT2 {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using numpy;
    using Python.Runtime;

    public static class Gpt2Dataset {
        internal static List<ndarray> LoadDataset(Gpt2Encoder encoder, string path, string pattern = "*") {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));
            var paths = new List<string>();
            if (Directory.Exists(path))
                paths.AddRange(Directory.EnumerateFiles(path, searchPattern: pattern, SearchOption.AllDirectories));
            else
                paths.Add(path);

            return LoadDataset(encoder, paths);
        }

        internal static List<ndarray> LoadDataset(Gpt2Encoder encoder, List<string> fileNames) {
            if (encoder == null) throw new ArgumentNullException(nameof(encoder));

            var tokenChunks = new List<ndarray>();
            foreach (string file in fileNames) {
                Debug.WriteLine($"Reading {file}");
                if (Path.GetExtension(file) == ".npz") {
                    // pre-encoded
                    dynamic npzObject = np.load(file);
                    var npz = npzObject.__enter__();
                    foreach (var item in npz.files)
                        tokenChunks.Add(npz[item]);
                    npzObject.__exit__();
                } else {
                    string rawText = File.ReadAllText(file);
                    if (String.IsNullOrWhiteSpace(rawText))
                        continue;
                    dynamic numpy = Py.Import("numpy");
                    PyObject tokens = numpy.stack(encoder.Encode(rawText));
                    tokenChunks.Add(ndarray.Wrap(tokens));
                }
            }

            return tokenChunks;
        }
    }
}
