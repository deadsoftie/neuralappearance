# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import neuralnetworks as nn
import numpy as np
import slangpy as spy
from util.write_model_to_buffer import write_models_to_buffer

from .aux_decoder import AuxDecoder
from .bsdf_decoder import BsdfDecoder
from .encoder import Encoder
from .latent_texture import LatentTexture
from .sampler import Sampler

if TYPE_CHECKING:
    from datagen import ReferenceMaterials

TrainingStatus = Literal['Initialized', 'Training', 'Done']


@dataclass
class InstancedComponent[T: nn.IModel]:
    """A set of candidate models tracked together during training."""

    ids: list[int]
    instances: list[T]
    status: TrainingStatus

    def active_training_models(self) -> list[nn.IModel]:
        if self.status != 'Training':
            return []
        return list(self.instances)

    def checkpoint_instance(self, best_instance_index: int) -> nn.IModel | None:
        if isinstance(self.instances[0], LatentTexture):
            # Always return the single latent texture instance.
            return self.instances[0]
        if self.status == 'Initialized':
            # Skip components that haven't been trained yet.
            return None
        if len(self.instances) == 1:
            # Only one remaining instance, use that one.
            return self.instances[0]

        # We're still training and pruning these instances, pick one based on
        # the current training loss.
        return self.instances[best_instance_index]

    def prune(self, indices) -> bool:
        if self.status != 'Training' or len(self.instances) == 1:
            return False

        # Preserve original IDs for logs/checkpoints while compacting the
        # runtime arrays to the loss-ranked subset.
        self.ids = [self.ids[i] for i in indices]
        self.instances = [self.instances[i] for i in indices]
        return True

    def start_training(self) -> int | None:
        self.status = 'Training'
        if isinstance(self.instances[0], LatentTexture):
            # Keep a single instance, but make sure it is optimizable. This
            # might allocate additional buffers to hold gradients.
            self.instances[0].make_optimizable()
            return None

        # Later phases may have a different candidate set from the phase that
        # just finished; the caller resets global instance bookkeeping to it.
        return len(self.instances)


@dataclass
class NeuralModelCheckpoint:
    num_mip_levels: int
    latent_texture: LatentTexture
    decoder: BsdfDecoder
    encoder: Encoder | None
    sampler: Sampler | None
    aux: AuxDecoder | None

    def components(self) -> list[tuple[str, nn.IModel | None]]:
        return [
            ('latents', self.latent_texture),
            ('decoder', self.decoder),
            ('encoder', self.encoder),
            ('sampler', self.sampler),
            ('aux', self.aux),
        ]


class NeuralModel:
    def __init__(
        self,
        module: spy.Module,
        config: dict,
        reference_materials: ReferenceMaterials,
    ):
        self.module = module
        self.config = config

        # Internally, we train potentially multiple "instances" of individual
        # neural components in parallel to converge to a good local optimum with
        # higher probability.
        self.num_instances = self.config['training']['instance_schedule'][0]
        self.instance_ids = list(range(self.num_instances))

        # The neural network library uses ``np.random`` to initialize the MLP
        # parameters. Together with the ``DeterministicOptimizer``, this ensures
        # reproducible runs.
        np.random.seed(self.config['model']['seed'])

        # Model initialization ...
        self.latent_texture: InstancedComponent[LatentTexture]
        self.decoder: InstancedComponent[BsdfDecoder]
        self.encoder: InstancedComponent[Encoder] | None = None
        self.sampler: InstancedComponent[Sampler] | None = None
        self.aux: InstancedComponent[AuxDecoder] | None = None

        # The latent texture contains a compressed, non-interpretable
        # representation of all spatially-varying material properties.
        latent_texture = LatentTexture.from_reference_materials(
            reference_materials,
            self.config['model']['latents']['num_mip_levels'],
            self.config['model']['latents']['num_channels'],
            optimizable=False,
        )
        latent_texture.initialize(self.module)

        # Only a single instance exists for the latent texture.
        self.latent_texture = InstancedComponent[LatentTexture](
            ids=[0],
            instances=[latent_texture],
            status='Training',
        )

        # The main neural component for evaluating the angular BSDF. It decodes
        # the latent code + the pair of incident and outgoing directions into
        # the BSDF reflectance value.
        self.decoder = self._create_instanced_component(
            lambda _: BsdfDecoder(
                nn.Real.half,
                self.config['model']['latents'],
                self.config['model']['decoder'],
            ),
            status='Training',
            name_prefix='BsdfDecoder',
        )

        if self.config['model'].get('encoder') is not None:
            num_inputs = reference_materials.num_encoder_inputs

            # The encoder learns the mapping from the set of reference material
            # attributes into a smaller number of latent channels. It only needs
            # to exist during training itself and its output will be baked into
            # latent textures before running inference.
            self.encoder = self._create_instanced_component(
                lambda _: Encoder(
                    nn.Real.half,
                    num_inputs,
                    self.config['model']['latents'],
                    self.config['model']['encoder'],
                ),
                status='Training',
                name_prefix='Encoder',
            )

            # In this case, the latent texture will not be optimized directly.
            self.latent_texture.status = 'Initialized'

        # The importance sampler, which is (optionally) trained in a separate
        # training phase.
        if self.config['model'].get('sampler') is not None:
            self.sampler = self._create_instanced_component(
                lambda _: Sampler(
                    nn.Real.half,
                    self.config['model']['latents'],
                    self.config['model']['sampler'],
                ),
                status='Initialized',
            )

        # The auxiliary output decoder, which is (optionally) trained in a
        # separate training phase.
        if self.config['model'].get('aux') is not None:
            self.aux = self._create_instanced_component(
                lambda _: AuxDecoder(
                    nn.Real.half,
                    self.config['model']['latents'],
                    self.config['model']['aux'],
                ),
                status='Initialized',
            )

    def load_checkpoint(
        self,
        model_path: str | Path,
    ) -> None:
        """Load model components from a checkpoint ``model.json`` file."""
        model_path = Path(model_path)
        assert model_path.is_file(), f'Checkpoint model file "{model_path}" does not exist.'

        with open(model_path) as f:
            ckpt_config = json.load(f)

        for name, component in self.components:
            if component is None:
                continue
            # The saved Slang type includes array sizes and nested model types,
            # so it detects incompatible checkpoint architectures.
            type_name = ckpt_config[name].pop('_type_name', None)
            if type_name is None:
                continue

            # Drop all but one instance.
            instance = component.instances[0]
            component.ids = [0]
            component.instances = [instance]
            # Mark this component as trained.
            component.status = 'Done'

            assert instance.type_name == type_name, (
                'Slang type mismatch between the neural model and the checkpoint.'
            )

            instance.load_checkpoint_images([name], model_path.parent)
            params = ckpt_config[name].pop('_params', None)
            if params is not None:
                instance.load_checkpoint_params(params)

    def _create_instanced_component[T: nn.IModel](
        self,
        factory: Callable[[int], T],
        status: TrainingStatus,
        name_prefix: str | None = None,
    ) -> InstancedComponent[T]:
        instances: list[T] = []
        for i in range(self.num_instances):
            instance = factory(i)
            if name_prefix is not None:
                instance.name = f'{name_prefix}[{i}]'
            instance.initialize(self.module)
            instances.append(instance)

        return InstancedComponent[T](
            ids=self.instance_ids.copy(),
            instances=instances,
            status=status,
        )

    @property
    def components(
        self,
    ) -> list[tuple[str, InstancedComponent[Any] | None]]:
        return [
            ('latents', self.latent_texture),
            ('decoder', self.decoder),
            ('encoder', self.encoder),
            ('sampler', self.sampler),
            ('aux', self.aux),
        ]

    @property
    def num_mip_levels(self) -> int:
        return self.config['model']['latents']['num_mip_levels']

    def active_training_models(self) -> list[nn.IModel]:
        result = []
        for _, c in self.components:
            if c is not None:
                result.extend(c.active_training_models())
        return result

    def write_to_buffer(self) -> None:
        # Training kernels receive one buffer for every set of candidate models.
        self.latent_texture_buffer = write_models_to_buffer(
            self.module, self.latent_texture.instances
        )
        self.decoder_buffer = write_models_to_buffer(self.module, self.decoder.instances)
        if self.encoder:
            self.encoder_buffer = write_models_to_buffer(self.module, self.encoder.instances)
        if self.sampler:
            self.sampler_buffer = write_models_to_buffer(self.module, self.sampler.instances)
        if self.aux:
            self.aux_buffer = write_models_to_buffer(self.module, self.aux.instances)

    def get_checkpoint(self, best_instance_index: int = 0) -> NeuralModelCheckpoint:
        """Extract the selected instance of every checkpoint component."""
        checkpoint_components: list[Any] = []
        for _, c in self.components:
            if c is None:
                # Skip components that don't exist.
                ckpt_c = None
            else:
                ckpt_c = c.checkpoint_instance(best_instance_index)
            checkpoint_components.append(ckpt_c)

        return NeuralModelCheckpoint(self.num_mip_levels, *checkpoint_components)

    def prune_instances(
        self, num_instances_to_keep: int, instance_losses: np.ndarray
    ) -> np.ndarray:
        """Prune every model component to the requested instance count."""
        assert num_instances_to_keep < self.num_instances
        assert len(instance_losses) == self.num_instances
        # One loss vector ranks every component participating in the current
        # phase. Pruning all training components with the same indices keeps
        # coupled encoder/decoder candidates aligned.
        sorted_indices = np.argsort(instance_losses)[:num_instances_to_keep]

        pruned_ids: list[int] | None = None
        for _, c in self.components:
            if c is None:
                continue

            if c.prune(sorted_indices):
                pruned_ids = c.ids

        if pruned_ids is not None:
            self.num_instances = len(pruned_ids)
            self.instance_ids = pruned_ids

        print(
            f'Pruning to {num_instances_to_keep} instances, remaining IDs: {self.instance_ids}.'
            if len(self.instance_ids) > 1
            else f'Pruning to 1 instance, remaining ID: {self.instance_ids[0]}.',
            flush=True,
        )
        return instance_losses[sorted_indices]

    def start_training[T: nn.IModel](
        self,
        component: InstancedComponent[T],
    ) -> None:
        """Start training ``component`` and reset the shared instance state."""

        num_instances = component.start_training()
        if num_instances is not None:
            # Reset number of instances.
            self.num_instances = num_instances
            self.instance_ids = component.ids.copy()
