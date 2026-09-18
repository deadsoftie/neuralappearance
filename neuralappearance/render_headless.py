"""Render an RTNAM checkpoint's neural and reference materials to PNG, no window.

Same scene/material/path-tracer setup as visualizer.py, minus the three
AppWindows (render/BSDF-plot/UV). Those only handle blit-to-swapchain and
presentation; HybridPathTracer.render() writes into a plain spy.Tensor
regardless, so this drives the same render calls and reads the tensor back
via falcor2's save_image() instead of presenting it.

Usage (same convention as train.py/visualizer.py: run from the
neuralappearance/ submodule root with the neuralappearance venv active):

    python neuralappearance/render_headless.py --jobs .jobs --spp 512 --out renders/

Renders every material in the checkpoint by default; pass --material to
pick one. --mip-level -1 (default) uses footprint-based dynamic level
selection, matching the paper's representative real-usage case rather than
visualizer.py's own fixed-level-0 startup default.
"""

import argparse
import copy
import time
from pathlib import Path

import falcor2 as f2
import slangpy as spy
from datagen import ReferenceMaterials
from falcor2.editor.utils import save_image
from model import NeuralModel, NeuralModelCheckpoint
from neuralnetworks import utils as nn_utils
from rendering.hybrid.hybrid_pathtracer import HybridPathTracer
from rendering.hybrid.scene_material_setup import prepare_scene_material
from rendering.neural_material import NeuralMaterial
from rendering.pathtracer_helpers import create_testscene
from util import get_default_asset_paths
from util.config import load_config

_FALCOR2_ROOT = Path(__file__).resolve().parent.parent / 'external' / 'falcor2'


def _resolve_checkpoint_paths(checkpoint: str) -> tuple[Path, Path]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_dir = checkpoint_path
        model_path = checkpoint_dir / 'model.json'
    else:
        model_path = checkpoint_path
        checkpoint_dir = model_path.parent

    config_path = checkpoint_dir / 'config.json'
    if not model_path.is_file():
        raise FileNotFoundError(f'Checkpoint model not found: {model_path}')
    if not config_path.is_file():
        raise FileNotFoundError(f'Checkpoint config not found: {config_path}')
    return config_path, model_path


def find_latest_checkpoint(path: str | Path) -> str:
    """Find the latest checkpoint from a job folder or jobs folder."""
    jobs_dir = Path(path)
    if (jobs_dir / 'checkpoints').is_dir():
        job_dirs = [jobs_dir]
    else:
        job_dirs = sorted(
            [d for d in jobs_dir.iterdir() if d.is_dir() and (d / 'checkpoints').is_dir()],
            key=lambda d: d.name,
        )
    if not job_dirs:
        raise FileNotFoundError(f'No jobs with checkpoints found in {jobs_dir}')
    latest_job = job_dirs[-1]
    ckpt_dirs = sorted(
        [d for d in (latest_job / 'checkpoints').iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f'No checkpoints found in {latest_job / "checkpoints"}')
    latest_ckpt = ckpt_dirs[-1]
    print(f'Using checkpoint: {latest_ckpt}')
    return str(latest_ckpt)


def _get_scene_camera(scene: f2.Scene) -> f2.Camera:
    for component in scene.components:
        if isinstance(component, f2.Camera):
            return component
    raise RuntimeError('No Camera component found in render scene')


class GpuAccumulator:
    """Progressively average one GPU image stream without CPU readback."""

    def __init__(self, device: spy.Device, width: int, height: int, utils_module: spy.Module):
        self._utils_module = utils_module
        self.tensor = spy.Tensor.empty(device, shape=(height, width), dtype=spy.float4)
        self.reset()

    def reset(self) -> None:
        self._utils_module.accumulator_reset(self.tensor)

    def update_and_output(self, input: spy.Tensor, output: spy.Tensor) -> None:
        if input is output:
            self._utils_module.accumulator_update_and_output_inplace(input, self.tensor)
        else:
            self._utils_module.accumulator_update_and_output(input, self.tensor, output)


class HeadlessRenderer:
    """Loads one checkpoint; renders its materials' neural/reference BSDFs to PNG."""

    def __init__(self, checkpoint: str, assets_paths: str, width: int, height: int):
        self.width = width
        self.height = height

        # Same device setup as visualizer.py (NeuralMaterialVisualizer.__init__):
        # the generated neural-material Slang needs the same include paths and
        # bindless buffer count, so this can't go through falcor2's generic
        # create_device() helper.
        self.device = spy.Device(
            type=spy.DeviceType.vulkan,
            enable_debug_layers=False,
            compiler_options={
                'include_paths': [
                    _FALCOR2_ROOT / 'slang',
                    Path(__file__).parent,
                    spy.SHADER_PATH,
                    *nn_utils.slang_include_paths(),
                ],
                'disable_warnings': [
                    '41018',  # returning without initializing out parameter 'value'
                    '41016',  # use of uninitialized variable
                    '41021',  # default initializer will not initialize field
                    '41035',  # possible use of uninitialized variable
                ],
            },
            bindless_options=spy.BindlessDesc(buffer_count=262144),
            enable_cuda_interop=False,
            enable_print=True,
            enable_hot_reload=False,
        )

        config_path, model_path = _resolve_checkpoint_paths(checkpoint)
        checkpoint_config = load_config(config_path)
        model_config = copy.deepcopy(checkpoint_config)

        self.checkpoint_reference_materials = ReferenceMaterials.from_config(model_config)
        training_module = spy.Module.load_from_file(self.device, 'training/train.slang')
        neural_model = NeuralModel(
            training_module,
            model_config,
            self.checkpoint_reference_materials,
        )
        neural_model.load_checkpoint(model_path)
        self.neural_model: NeuralModelCheckpoint = neural_model.get_checkpoint()

        self.ref_scene = create_testscene(self.device)
        self.reference_materials = ReferenceMaterials.from_falcor2(
            self.device,
            copy.deepcopy(checkpoint_config),
            assets_paths,
            scene=self.ref_scene,
        )
        assert len(self.reference_materials) == len(self.checkpoint_reference_materials)

        self.neural_scene = create_testscene(self.device)
        self.neural_material = self.neural_scene.create_material(NeuralMaterial)
        self.neural_material.name = 'neural_material'

        config_max_depth = model_config['checkpoints']['rendering'].get('max_depth', 10)

        self.hybrid_pt_neural = HybridPathTracer(self.device)
        self.hybrid_pt_neural.max_depth = config_max_depth
        self.hybrid_pt_neural.scene = self.neural_scene

        self.hybrid_pt_ref = HybridPathTracer(self.device)
        self.hybrid_pt_ref.max_depth = config_max_depth
        self.hybrid_pt_ref.scene = self.ref_scene

        self.camera = _get_scene_camera(self.ref_scene)
        self.camera.width = width
        self.camera.height = height

        utils_module = spy.Module.load_from_file(self.device, 'falcor2/utils.slang')
        self.neural_tensor = spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4)
        self.ref_tensor = spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4)
        self.neural_accum = GpuAccumulator(self.device, width, height, utils_module)
        self.ref_accum = GpuAccumulator(self.device, width, height, utils_module)

    def render_material(
        self,
        material_idx: int,
        spp: int,
        mip_level: int,
        views: str,
        out_dir: Path,
    ) -> None:
        ref_mat = self.reference_materials[material_idx]
        checkpoint_mat = self.checkpoint_reference_materials[material_idx]

        prepare_scene_material(self.ref_scene, ref_mat.falcor2_handle)
        self.neural_material.configure(self.device, self.neural_model, checkpoint_mat.id)
        prepare_scene_material(self.neural_scene, self.neural_material)

        self.hybrid_pt_neural.set_target_material(None, 0)
        self.hybrid_pt_ref.set_target_material(None, ref_mat.texture_resolution)

        label = f'mat{material_idx}-id{ref_mat.id}-{ref_mat.type}'

        # Flushing periodically (rather than only once at the end) keeps the queued command count bounded
        flush_interval = 128

        if views in ('both', 'neural'):
            self.neural_accum.reset()
            for frame in range(spp):
                self.hybrid_pt_neural.render(self.camera, self.neural_tensor, frame, mip_level, True)
                self.neural_accum.update_and_output(self.neural_tensor, self.neural_tensor)
                if (frame + 1) % flush_interval == 0:
                    self.device.wait()
            self.device.wait()
            save_image(self.neural_tensor, out_dir / f'{label}_neural_{spp}spp.png')

        if views in ('both', 'reference'):
            self.ref_accum.reset()
            for frame in range(spp):
                self.hybrid_pt_ref.render(self.camera, self.ref_tensor, frame, mip_level, False)
                self.ref_accum.update_and_output(self.ref_tensor, self.ref_tensor)
                if (frame + 1) % flush_interval == 0:
                    self.device.wait()
            self.device.wait()
            save_image(self.ref_tensor, out_dir / f'{label}_reference_{spp}spp.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Headless render of RTNAM checkpoint materials (neural vs reference) to PNG.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--checkpoint', type=str, help='Path to the neural material checkpoint directory'
    )
    group.add_argument(
        '--jobs',
        type=str,
        help='Path to the jobs folder; the latest job and its latest checkpoint will be used',
    )
    parser.add_argument(
        '-a',
        '--assets-paths',
        default=get_default_asset_paths(),
        help='Semicolon-separated asset roots. Defaults to the repo root.',
    )
    parser.add_argument(
        '--material',
        type=int,
        default=None,
        help='Material index to render. Omit to render every material in the checkpoint.',
    )
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument(
        '--spp',
        type=int,
        default=256,
        help='Samples accumulated per image. RTNAM paper used 8192 for converged references.',
    )
    parser.add_argument(
        '--mip-level',
        type=int,
        default=-1,
        help='-1 = dynamic footprint-based level selection (default), 0..N = fixed level.',
    )
    parser.add_argument(
        '--views',
        choices=['both', 'neural', 'reference'],
        default='both',
    )
    parser.add_argument('--out', type=str, default='render_headless_out')
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint else find_latest_checkpoint(args.jobs)

    renderer = HeadlessRenderer(checkpoint, args.assets_paths, args.width, args.height)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    material_indices = (
        [args.material] if args.material is not None else range(len(renderer.reference_materials))
    )

    for idx in material_indices:
        t0 = time.perf_counter()
        renderer.render_material(idx, args.spp, args.mip_level, args.views, out_dir)
        elapsed = time.perf_counter() - t0
        print(f'material {idx}: {elapsed:.1f}s for {args.spp} spp @ {args.width}x{args.height}')
