# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from slangpy import Module, Struct, Tensor
from slangpy.reflection import SlangType

from .real_array import ArrayKind, RealArray

TypeLike = str | SlangType | Struct | RealArray

if TYPE_CHECKING:
    from ..components.model_chain import ChainedModelPair


class ModelError(Exception):
    """Error raised while configuring, initializing, or traversing a model."""

    pass


class ModelRecordKind(enum.IntEnum):
    """Reserved debug records automatically available on recordable models."""

    Input = 0
    Output = enum.auto()
    InputGrad = enum.auto()
    OutputGrad = enum.auto()
    Count = enum.auto()


class ModelRecord:
    """Description of one value captured by the model debug recorder."""

    def __init__(self, index: int, name: str, data_type_name: str):
        self.index = index
        self.name = name
        self.data_type_name = data_type_name


class IModel:
    """Python representation of a Slang type implementing ``IModel``.

    Construction is intentionally split into two phases. ``__init__`` stores
    configuration, while ``initialize()`` receives a loaded Slang module and
    an input type, runs model-specific setup, and checks the matching Slang
    ``forward()`` signature. Methods that inspect types, parameters, or shader
    data generally require initialization first.

    Subclassing checklist
    ---------------------
    Every subclass must:

    * call ``super().__init__()`` from its constructor; and
    * implement ``type_name``, naming its corresponding Slang type.

    Override the following hooks only when the model needs them:

    * ``model_init()`` to validate the reflected input type, resolve ``Auto``
      values, allocate state, or initialize child models;
    * ``resolve_input_type()`` to allow ``initialize(module)`` without an
      explicit input type;
    * ``model_params()`` for trainable tensors owned directly by this model;
    * ``model_data()`` for fields of the corresponding Slang struct;
    * ``children()`` and ``child_name()`` for a composite model; and
    * the checkpoint methods when the model owns state requiring custom
      serialization.

    ``initialize()``, ``parameters()``, ``components()``, and ``get_this()``
    implement the framework lifecycle and normally should not
    be overridden.
    """

    next_model_id = 0

    def __init__(self):
        super().__init__()

        self._initialized = False
        self.parent: IModel | None = None
        self._input_type: SlangType
        self._output_type: SlangType
        self.debuggable = False
        self.debug_records: list[ModelRecord] = []
        self.record_count = ModelRecordKind.Count.value
        self.model_id = -1
        self.name: str | None = None

    # Public lifecycle and inspection.

    def initialize(self, module: Module, input_type: TypeLike | None = None):
        """Initialize the model, allocate its state, and check its Slang types.

        ``module`` must contain the Slang type returned by ``type_name`` and
        all types on which it depends. The calling Slang module will usually
        need ``import neural_networks;`` plus imports for application-specific
        model types.

        ``input_type`` describes the value passed to the Slang ``forward()``
        method. It is used for type checking and resolving ``Auto`` parameters,
        and may be a reflected ``SlangType``, a type-name string, a SlangPy
        ``Struct``, or a ``RealArray``.

        ``input_type`` may be omitted when ``resolve_input_type()`` can derive
        it from the model configuration. Applications should call this method,
        not the ``model_init()`` hook.
        """

        # Give a focused error when a subclass omitted super().__init__().
        if not hasattr(self, '_initialized'):
            raise RuntimeError(
                'The constructor of this model was never called. '
                'You likely forgot a super().__init__() call in '
                f'class {type(self).__name__} or one of its superclasses'
            )
        if input_type is None:
            input_type = self.resolve_input_type(module)
        if isinstance(input_type, RealArray):
            input_type = input_type.name()
        if isinstance(input_type, str):
            input_type = self._lookup_mandatory_type(module, input_type)
        if isinstance(input_type, Struct):
            input_type = input_type.struct
        if input_type is None:
            self.model_error(
                'initialize() cannot proceed: No input_type was provided, and '
                "the model can't resolve it by itself, either because the model "
                'does not implement it or because some parameters are set to Auto.'
            )

        try:
            self.model_init(module, input_type)
        except ModelError:
            raise
        except Exception as e:
            self.model_error(f'{type(e).__name__}: {e}')

        self._input_type = input_type
        self._initialized = True

        type_name = self.type_name
        model_type = self._lookup_mandatory_type(module, type_name)

        if len(type_name) > 50:
            short_type_name = type_name[:47] + '...'
            full_type_msg = f'. The full type name was {type_name}'
        else:
            short_type_name = type_name
            full_type_msg = ''

        forward = module.layout.find_function_by_name_in_type(model_type, 'forward')
        if forward is None:
            self.model_error(
                f'Looking up method forward() in type {short_type_name} failed. Make sure the type '
                f'implements the IModel interface{full_type_msg}'
            )

        # Specializing an overloaded forward() can trigger a Slang reflection
        # bug. For overloaded methods, identify the return type through the
        # matching IModel witness instead. Non-overloaded methods use the normal
        # specialization path below.
        if forward.is_overloaded:
            return_types = {
                f.return_type.full_name for f in forward.overloads if f.return_type is not None
            }
            candidates = []
            for candidate in return_types:
                witness_name = (
                    f'impl::return_type_witness<{input_type.full_name}, {candidate}, {type_name}>'
                )
                witness = module.layout.find_function_by_name(witness_name)
                if witness is not None:
                    candidates.append(candidate)
            if len(candidates) > 1:
                self.model_error(
                    f'Found multiple matching overloads for method forward({input_type.full_name}) in type {short_type_name}, '
                    f'and the return type is ambiguous (found {candidates}). Make sure there is only one forward() '
                    f'implementation for each input type.{full_type_msg}'
                )
            elif len(candidates) == 0:
                self.model_error(
                    f'Could not find a matching overload for method forward({input_type.full_name}) in type {short_type_name}. '
                    'The most common cause is that the output of the previous model is not compatible '
                    f'with the input expected by the next model, e.g. due to mismatched dimensions '
                    f'or element precision{full_type_msg}'
                )
            else:
                self._output_type = self._lookup_mandatory_type(module, candidates[0])
        else:
            specialized = forward.specialize_with_arg_types([input_type])
            if specialized is None:
                self.model_error(
                    f'Could not find a matching overload for method forward({input_type.full_name}) in type {short_type_name}. '
                    'The most common cause is that the output of the previous model is not compatible '
                    f'with the input expected by the next model, e.g. due to mismatched dimensions '
                    f'or element precision{full_type_msg}'
                )
            if specialized.return_type is None:
                self.model_error(
                    f'The method forward({input_type.full_name}) in type {short_type_name} does not return a value. '
                    f'Make sure the model conforms to the IModel interface{full_type_msg}'
                )

            self._output_type = specialized.return_type

        witness_name = f'impl::recordable_witness<{self._input_type.full_name}, {self._output_type.full_name}, {self.type_name}>'
        witness = module.find_function(witness_name)
        if witness is None:
            try:
                in_arr = RealArray.from_slangtype(self.input_type)
                out_arr = RealArray.from_slangtype(self.output_type)
                if (
                    in_arr.kind == ArrayKind.coopvec
                    and out_arr.kind == ArrayKind.coopvec
                    and in_arr.dtype == out_arr.dtype
                ):
                    witness_name = f'impl::recordable_witness<{in_arr.dtype}, {in_arr.length}, {out_arr.length}, {self.type_name}>'
                    witness = module.find_function(witness_name)
            except Exception:
                pass

        if witness is not None:
            self.debuggable = True
            self.debug_records.extend(
                [
                    ModelRecord(ModelRecordKind.Input, 'Input', self._input_type.full_name),
                    ModelRecord(ModelRecordKind.Output, 'Output', self._output_type.full_name),
                    ModelRecord(
                        ModelRecordKind.InputGrad,
                        'InputGrad',
                        self._input_type.derivative.full_name,
                    ),
                    ModelRecord(
                        ModelRecordKind.OutputGrad,
                        'OutputGrad',
                        self._output_type.derivative.full_name,
                    ),
                ]
            )
            self.model_id = IModel.next_model_id
            IModel.next_model_id += 1
        else:
            if self.debug_records:
                self.model_error(
                    'Model is not recordable, but debug records were registered. '
                    'Make sure the model implements the IRecordableModel interface.'
                )

    def get_this(self) -> dict[str, Any]:
        """Return the complete SlangPy representation of this model.

        This framework method combines ``type_name``, ``model_data()``,
        and debug metadata. Call it when embedding a model in another model's
        data or passing a model to a Slang function; do not override it.
        """
        self.check_initialized()
        data = {'_type': self.type_name} | self.model_data()
        if self.debuggable:
            data = data | {'model_id': self.model_id}
        return data

    @property
    def input_type(self) -> SlangType:
        """Return the reflected input type of the Slang ``forward()`` method."""
        self.check_initialized()
        return self._input_type

    @property
    def output_type(self) -> SlangType:
        """Return the reflected Slang ``forward()`` output type."""
        self.check_initialized()
        return self._output_type

    def components(self) -> list[IModel]:
        """Return this model and all descendants in traversal order.

        This framework method follows ``children()`` recursively.
        """
        self.check_initialized()
        return list(self._component_iter())

    def parameters(self) -> list[Tensor]:
        """Return all tensors that should receive optimizer state.

        This framework method combines ``model_params()`` from every model in
        ``components()``.
        """
        self.check_initialized()
        result = []
        for c in self._component_iter():
            result += c.model_params()
        return result

    def register_debug_record(self, record_index: int, record_name: str, data_type_name: str):
        """Register a value recorded by this model's Slang implementation.

        Call this before creating a model recorder; do not override it. The
        Slang type must implement ``IRecordableModel`` for recording to be
        available.
        """
        if self._initialized and not self.debuggable:
            self.model_error(
                'The slang type of this model must implement IRecordableModel to use debug records'
            )

        if record_index < 0:
            self.model_error(f'Invalid index {record_index} for record {record_name}')
        if record_index < ModelRecordKind.Count.value:
            self.model_error(f'Record {record_name} is using a reserved index {record_index}')

        self.record_count = max(self.record_count, record_index + 1)
        self.debug_records.append(ModelRecord(record_index, record_name, data_type_name))

    # Subclass extension points.

    @property
    def type_name(self) -> str:
        """Return the name of the Slang type implementing this model.

        Required subclass hook. Include generic arguments and nested model
        types so the name identifies the concrete Slang specialization.
        """
        self.model_error('type_name is not implemented')

    def model_init(self, module: Module, input_type: SlangType):
        """Perform model-specific initialization.

        Optional subclass hook called by ``initialize()`` with a reflected
        input type. Override it to validate or inspect that type, resolve
        ``Auto`` configuration, allocate parameters, or initialize children.
        Do not call this hook directly.
        """

    def resolve_input_type(self, module: Module) -> TypeLike | None:
        """Derive this model's input type from its configuration.

        Optional subclass hook used when ``initialize()`` receives no
        ``input_type``. Return ``None`` when an upstream caller or model chain
        must provide the type, typically because some dimensions are ``Auto``.
        """
        return None

    def model_params(self) -> list[Tensor]:
        """Return trainable tensors owned directly by this model.

        Optional subclass hook. Do not include child parameters; the default
        traversal collects those through ``children()``.
        """
        return []

    def model_data(self) -> dict[str, Any]:
        """Return fields used to construct this model's Slang value.

        Optional subclass hook. Dictionary keys correspond to instance fields
        on the Slang struct named by ``type_name``. Values must be
        representable by SlangPy; nested models are normally supplied using
        their ``get_this()`` result.

        ``get_this()`` adds the ``_type`` field and optional debugging data,
        so implementations only return fields declared by their own Slang
        struct.
        """
        return {}

    def children(self) -> list[IModel]:
        """Return this model's immediate child models.

        Optional subclass hook for composite models. The result drives model
        traversal, checkpoint recursion, parameter collection, and debugging.
        """
        return []

    def child_name(self, child: IModel) -> str | None:
        """Return a stable, human-readable name for an immediate child.

        Optional subclass hook for composite models. Names are used in error
        paths and as keys in hierarchical checkpoints. Return ``None`` when a
        child does not need its own path segment.
        """
        return None

    # Checkpoint extension points. The default implementations recurse through
    # children; models override them when they own serializable state.

    def save_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ) -> list[tuple[list[str], str]]:
        """Save image-based state and return ``(description, filename)`` pairs.

        Optional subclass hook. ``desc`` is the current checkpoint path split
        into segments. The default implementation recursively saves children,
        using ``child_name()`` to extend that path.
        """

        def expand_chain(x: IModel) -> list[IModel]:
            # Flatten ModelChain hierarchy if it's in the way.
            if type(x).__name__ == 'ChainedModelPair':
                x = cast('ChainedModelPair', x)
                return expand_chain(x.first) + expand_chain(x.second)
            else:
                return [x]

        result: list[tuple[list[str], str]] = []
        for child in self.children():
            for c in expand_chain(child):
                child_name = self.child_name(c)

                desc_c = desc.copy()
                if child_name is not None:
                    desc_c.append(child_name)

                result += c.save_checkpoint_images(desc_c, folder)
        return result

    def load_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ) -> None:
        """Load image-based state previously written for ``desc``.

        Optional subclass hook paired with ``save_checkpoint_images()``. The
        default implementation recursively loads children.
        """

        def expand_chain(x: IModel) -> list[IModel]:
            # Flatten ModelChain hierarchy if it's in the way.
            if type(x).__name__ == 'ChainedModelPair':
                x = cast('ChainedModelPair', x)
                return expand_chain(x.first) + expand_chain(x.second)
            else:
                return [x]

        for child in self.children():
            for c in expand_chain(child):
                child_name = self.child_name(c)

                desc_c = desc.copy()
                if child_name is not None:
                    desc_c.append(child_name)

                c.load_checkpoint_images(desc_c, folder)

    def save_checkpoint_params(
        self,
        desc: list[str],
    ) -> list[tuple[list[str], dict]]:
        """Return JSON-compatible parameter records for this checkpoint path.

        Optional subclass hook for state not stored as images. The default
        implementation recursively gathers records from children.
        """

        def expand_chain(x: IModel) -> list[IModel]:
            # Flatten ModelChain hierarchy if it's in the way.
            if type(x).__name__ == 'ChainedModelPair':
                x = cast('ChainedModelPair', x)
                return expand_chain(x.first) + expand_chain(x.second)
            else:
                return [x]

        result: list[tuple[list[str], dict]] = []
        for child in self.children():
            for c in expand_chain(child):
                child_name = self.child_name(c)

                desc_c = desc.copy()
                if child_name is not None:
                    desc_c.append(child_name)

                result += c.save_checkpoint_params(desc_c)
        return result

    def load_checkpoint_params(
        self,
        params: dict,
    ) -> None:
        """Restore parameter records produced by ``save_checkpoint_params()``.

        Optional subclass hook. The default implementation recursively passes
        each named subsection to the corresponding child.
        """

        def expand_chain(x: IModel) -> list[IModel]:
            # Flatten ModelChain hierarchy if it's in the way.
            if type(x).__name__ == 'ChainedModelPair':
                x = cast('ChainedModelPair', x)
                return expand_chain(x.first) + expand_chain(x.second)
            else:
                return [x]

        for child in self.children():
            for c in expand_chain(child):
                child_name = self.child_name(c)

                params_c = params.copy()
                if child_name is not None and child_name in params_c:
                    params_c = params[child_name]
                c.load_checkpoint_params(params_c)

    # Internal framework helpers.

    def set_parent(self, parent: IModel):
        """Set the parent used to build paths in ``model_error()``.

        Composite-model constructors call this when attaching children. This is
        a framework helper, not a subclass hook.
        """
        self.parent = parent

    def check_initialized(self):
        """Raise ``ModelError`` if this model has not been initialized."""
        if not self._initialized:
            self.model_error(
                'Model is uninitialized. Make sure to call .initialize() before using the model'
            )

    def model_error(self, msg: str) -> NoReturn:
        """Raise ``ModelError`` with this model's component path."""
        segments: list[str] = []
        child = self
        while child:
            child_name = type(child).__name__
            if child.parent:
                readable_name = child.parent.child_name(child)
                if readable_name is not None:
                    child_name = f'{readable_name}: {child_name}'
            segments = [child_name, *segments]
            child = child.parent

        component_name = type(self).__name__
        component_path = ' -> '.join(segments)
        raise ModelError(
            'Encountered an error while handling model component '
            f'{component_name} (with path {component_path}): {msg}'
        )

    def _lookup_mandatory_type(self, module: Module, name: str) -> SlangType:
        lookup = module.layout.find_type_by_name(name)

        if lookup is None:
            self.model_error(
                'Looking up slang type failed. This might be because of a missing import, or '
                'because of a type error. Try pasting the type name into the slang '
                f'module and check for compilation errors to help diagnose: {name}'
            )

        return lookup

    def _component_iter(self):
        yield self
        for c in self.children():
            yield from c._component_iter()
