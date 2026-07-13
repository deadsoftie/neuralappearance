# Neural Networks

This directory contains the small neural-network framework used by the material models in this
repository. Python constructs and initializes model graphs, while Slang implements their GPU
forward and backward passes. It includes the shared model interface, linear layers, activations,
type conversions, losses, optimizers, and optional debug recording.

The code is intentionally more general than the material model in `../model/`. It evolved for a
time as a standalone-style library, and an older relative of it also exists in the SlangPy
examples. The version here is the one used and maintained by neuralappearance; the copies should
not be assumed to have matching APIs or behavior.

## Python and Slang roles

Most features have a Python side and a Slang side:

- Python stores configuration, resolves `Auto` properties from reflected types, allocates tensors,
  and exposes model data to SlangPy.
- Slang defines the statically typed model and optimizer implementations that execute on the GPU.
- `neural_networks.slang` is the umbrella Slang module. Callers add this directory to the module
  search path with `slang_include_paths()` and use `import neural_networks;` in Slang.

`IModel` is the bridge between the two sides. A Python model's `type_name` identifies a Slang type
implementing `IModel<InputT, OutputT>`. During `initialize()`, Python reflects that type, checks its
`forward()` signature, allocates any model state, and prepares the value returned by `get_this()`
for SlangPy. See `basetypes/imodel.py` for the complete subclassing contract.

Optimizers follow the same split. `optimizers/optimizer.py` gathers parameters and owns the Python
lifecycle, while a Slang type implementing `IOptimizer<T>` performs the per-element update. A new
optimizer normally implements `get_type_name()` and `get_this()` in Python and the `State`,
`Batch`, `step()`, and `batch_step()` contract in Slang.

## Directory layout

- `basetypes/`: model contracts and shared descriptions of Slang scalar and array types.
- `components/`: reusable layers, activations, conversions, and sequential composition.
- `losses/`: differentiable loss functions consumed by the training kernels.
- `optimizers/`: optimizer dispatch, Adam, full-precision shadow parameters, and deterministic
  gradient reduction.
- `debug/`: opt-in capture of intermediate values and gradients.
- `utils.py`: Slang include paths and parameter-buffer merging.
