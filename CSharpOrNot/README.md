## Running the sample

This sample requires TensorFlow 1.14 due to a bug in `BatchNormalization` layer in 1.10.

1. Create new Conda environment named `tf-1.14`: `conda create -n tf-1.x python=3.7`
(`brew cask install miniconda` to get `conda command`)
1. Activate that environment: `conda activate tf-1.14`
1. Install TensorFlow 1.14: `python -m pip install "tensorflow==1.14.*"`
1. Navigate to this directory
1. Set environment variable `GRADIENT_PYTHON_ENVIRONMENT` to `conda:tf-1.14`
 (e.g. `set ... = ...` on Windows, `export ... = ...` on Linux and Mac)
1. Run `dotnet run --framework netcoreapp3.0 -- ui`

## Using the sample

1. Download & extract pretrained model from
[GitHub](https://github.com/losttech/Gradient-Samples/releases/download/csharp-or-not%2Fv1/csharp-or-not-v1.zip)
or train your own (instructions pending)
1. Launch CSharpOrNot (see above). A dialog will appear to load model weights.
1. Find weights `.index` or `.hdf5` file to load the model
1. Click on "Open File..." to load code
1. Select any source code file from your projects
1. After it is loaded, move cursor around and watch programming language detected
1. Click on "Open File..." again to load a different code file (no need to load model again)
