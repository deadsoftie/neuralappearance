# Neural Appearance

![Teaser](https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/featured_hud06a832d8d633d2c650140e8f6b3b1a8_417527_720x0_resize_q90_lanczos.jpg)

This is a research pipeline that converts spatially varying MaterialX and MDL materials into compact neural material models. Given a physically based reference material, it generates BSDF training samples on the GPU and learns latent textures together with small neural networks for BSDF evaluation, importance sampling, and optional auxiliary material signals.

The pipeline is built on [Slang](https://shader-slang.org/), [SlangPy](https://github.com/shader-slang/slangpy), and [falcor2](https://github.com/NVlabs/falcor2), with Python serving as the orchestration layer.

This repository accompanies the technical paper

**[Taming optimization variance in compact neural shading networks](https://research.nvidia.com/labs/rtr/publication/bitterli2026taming/)**<br>
*SIGGRAPH 2026 (Conference Track)*<br>
Benedikt Bitterli, Petrik Clarberg, Chris Cummings, Aaron Lefohn, Steve Marschner, Jan Novák, Fabrice Rousselle, Andrea Weidlich, Tizian Zeltner

Additional details can be found in

**[Real-Time Neural Appearance Models](https://research.nvidia.com/labs/rtr/neural_appearance_models/)**<br>
*Transactions on Graphics (Presented at SIGGRAPH 2024)*<br>
Tizian Zeltner, Fabrice Rousselle, Andrea Weidlich, Petrik Clarberg, Jan Novák, Benedikt Bitterli, Alex Evans, Tomáš Davidovič, Simon Kallweit, Aaron Lefohn

and

**[Bridging the Gap Between Offline and Real Time with Neural Materials](https://blog.selfshadow.com/publications/s2025-shading-course/)**<br>
*SIGGRAPH 2025 Course: Physically Based Shading in Theory and Practice*<br>
Andrea Weidlich

The work represented in this repository includes contributions from the authors of the publications above, Craig Kolb, Wessam Bahnassi, Yaobin Ouyang, Andrew Allan, and Alexey Bekin.


## Prerequisites

- Python 3.12 or newer
- Git with Git LFS enabled, required by the recursive SlangPy data submodule
- A Vulkan-capable NVIDIA GPU with Cooperative Vector (CoopVec) support and a current driver
- A C++ build environment supported by falcor2/SlangPy

On Windows, use Visual Studio 2022 or Visual Studio 2022 Build Tools with C++
build tools installed. The setup script prepares the matching falcor2 and
SlangPy dependencies for this release.

## Quickstart

From the root of the cloned repository, create and activate a local Python
environment, then run the platform setup script.

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
setup.bat
```

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
./setup.sh
```

Train the bundled sample material:

```bash
python neuralappearance/train.py -c configs/default.json -o .jobs/example
```

The default config trains the sample MaterialX material at
`assets/materials/FauxLeather.mtlx`.

## Training

The main entry point is:

```bash
python neuralappearance/train.py
```

Common arguments:

- `-c, --config`: training config path. Defaults to `configs/default.json`.
- `-o, --outfolder`: output folder. Defaults to a timestamped folder under `.jobs/`.
- `-a, --assets-paths`: semicolon-separated asset search roots. Defaults to the repo root.
- `-v, --view`: launch the visualizer after training completes.

Example with an explicit asset root:

```bash
python neuralappearance/train.py -c configs/default.json -a "D:/datasets/materials" -o .jobs/example
```

### Paper configurations

The teaser experiment from Figure 1 is represented by three derived configs:
[`single_instance_training_large.json`](configs/single_instance_training_large.json)
uses an overparameterized decoder that trains consistently,
[`single_instance_training_small.json`](configs/single_instance_training_small.json)
exposes the seed sensitivity of an undersized decoder, and
[`multi_instance_training_small.json`](configs/multi_instance_training_small.json)
shows how multi-instance training makes that same small decoder reliable without
increasing the training cost.

## Configs and Assets

Training behavior is controlled by JSON config files. Start with
`configs/default.json`, which documents the main training, model, data
generation, and checkpoint options.

The training inputs are one or more physically based reference materials in
MaterialX or MDL. They are evaluated directly on the GPU through falcor2's
material system, so no pre-generated dataset is required. The MaterialX backend
builds on the official MaterialX libraries and generates Slang shaders, while
the MDL backend compiles materials through NVIDIA's MDL SDK. Multiple materials
can be trained jointly, resulting in separate latent textures and one shared
neural model.

### Resolving material assets

Material paths in the config are relative to one of the roots supplied through
`--assets-paths`. The option accepts a semicolon-separated list and defaults to
the repository root. This makes the bundled MaterialX example available as
`assets/materials/FauxLeather.mtlx` without additional arguments, while external
material collections can live anywhere on the system.

A MaterialX entry identifies one `.mtlx` document:

```json
{
    "type": "mtlx",
    "path": "assets/materials/FauxLeather.mtlx"
}
```

falcor2 loads the document through its MaterialX backend and uses the asset
roots to resolve referenced documents and resources.

An MDL entry identifies a `.mdl` module and one material exported by it:

```json
{
    "type": "mdl",
    "path": "vMaterials_2/Wood/Wood_Tiles_Pine.mdl",
    "material": "Wood_Tiles_Pine_Mosaic"
}
```

The matching asset root becomes the MDL module search root. The relative path
above maps to the module `vMaterials_2::Wood::Wood_Tiles_Pine`; falcor2 then asks
the MDL SDK to compile its exported `Wood_Tiles_Pine_Mosaic` material. Imports,
textures, and other module resources are resolved from the same MDL hierarchy.

This example uses NVIDIA's vMaterials collection. Download and install the
current vMaterials 2.x package from
[NVIDIA's vMaterials page](https://developer.nvidia.com/vmaterials), then set
`--assets-paths` to the installed `mdl` folder. On Windows, this is commonly
`%USERPROFILE%\Documents\mdl`:

```cmd
python neuralappearance/train.py -c configs/default.json -a "%USERPROFILE%\Documents\mdl" -o .jobs/vmaterials
```

## Outputs

Training outputs are written to the selected job folder, usually under `.jobs/`.
Important files include:

- `summary.json`: job metadata and per-phase timing
- `output.log`: captured training log
- `loss_series.json`: recorded loss curves
- `loss_plots/`: rendered loss plots when enabled
- `checkpoints/`: saved model checkpoints and checkpoint artifacts

Checkpoint contents depend on the `checkpoints` section of the config.

## Visualization

Open the latest checkpoint from a jobs folder in the interactive visualizer:

```bash
python neuralappearance/visualizer.py --jobs .jobs
```

You can also pass a checkpoint directory explicitly:

```bash
python neuralappearance/visualizer.py --checkpoint .jobs/example/checkpoints/<checkpoint>
```

The visualizer accepts the same `-a, --assets-paths` option described above.
Pass the asset roots used during training so it can load the reference materials
for comparison.

Checkpoint folders are named by training iteration.

## Contributing

This project is currently not accepting contributions.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
