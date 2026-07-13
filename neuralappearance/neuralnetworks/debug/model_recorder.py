# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import json
import operator
import os
from itertools import chain
from typing import Any

import numpy as np
import slangpy as spy
from slangpy.core.function import FunctionNode

from ..basetypes import IModel
from ..components import ModelChain


class ModelRecorder:
    """Capture selected model values for offline inspection.

    Models opt in through ``IModel`` debug records. ``setup_function()`` enables
    Slang instrumentation, while ``dump_data()`` writes a JSON description and
    an NPZ containing parameters and recorded per-sample values.
    """

    def __init__(
        self,
        module: spy.Module,
        models: list[IModel],
        batch_size: int,
        out_folder: str,
        iteration: int,
    ):
        model_count = IModel.next_model_id
        record_base = np.zeros((1 + model_count,), dtype=np.int32)
        for m in models:
            for c in m.components():
                if c.debuggable:
                    record_base[c.model_id + 1] = c.record_count
        np.cumsum(record_base, out=record_base)

        record_count = record_base[-1]
        records = spy.Tensor.zeros(module.device, (record_count,), module['DebugRecord'])
        record_cursor = records.cursor()

        record_data: dict[tuple[int, int], spy.Tensor] = {}

        for component in chain.from_iterable(m.components() for m in models):
            if not component.debuggable:
                continue

            for record in component.debug_records:
                storable_type_name = record.data_type_name
                type = module.find_struct(storable_type_name)
                if type is None:
                    component.model_error(
                        f"Can't set up debug recording for {record.name} because I can't find type {storable_type_name}"
                    )

                # Buffers cannot store ``CoopVec`` directly, so use an
                # equivalent array type.
                if type.name == 'CoopVec':
                    type = module.find_struct('Array' + type.full_name[len('CoopVec') :])
                    assert type is not None

                buffer = spy.Tensor.zeros(module.device, (batch_size,), type)
                record_data[(component.model_id, record.index)] = buffer
                buffer_index = record_base[component.model_id] + record.index
                record_cursor[buffer_index]['entry_size'] = type.buffer_layout.stride
                record_cursor[buffer_index]['data'] = buffer.storage.descriptor_handle_rw

        record_cursor.apply()

        self.models = models
        self.batch_size = batch_size
        self.records = records
        self.record_base = spy.Tensor.from_numpy(module.device, record_base)
        self.record_data = record_data
        self.out_folder = out_folder
        self.iteration = iteration

    def type_to_json(self, refl: spy.TypeLayoutReflection, offset: int):
        """Convert a reflected buffer layout to the recorder's JSON schema."""

        children = []

        if refl.kind == spy.TypeReflection.Kind.array:
            for i in range(refl.type.element_count):
                children.append(
                    self.type_to_json(
                        refl.element_type_layout, offset + i * refl.element_type_layout.stride
                    )
                )
        elif refl.kind == spy.TypeReflection.Kind.vector:
            vector_names = ['x', 'y', 'z', 'w']
            for i, name in enumerate(vector_names[: refl.type.col_count]):
                children.append(
                    self.type_to_json(
                        refl.element_type_layout, offset + i * refl.element_type_layout.stride
                    )
                    | {'name': name}
                )
        elif refl.kind == spy.TypeReflection.Kind.matrix:
            pass
        elif refl.kind == spy.TypeReflection.Kind.scalar:
            pass
        elif refl.kind == spy.TypeReflection.Kind.struct:
            for field in refl.fields:
                children.append(
                    self.type_to_json(field.type_layout, offset + field.offset)
                    | {'name': field.name}
                )

        result = {'type': refl.name, 'offset': offset, 'size': refl.size}
        if children:
            result['children'] = children

        return result

    def setup_function(self, func: FunctionNode) -> FunctionNode:
        """Bind recorder buffers and enable recording for a SlangPy function."""

        func = func.set(
            {
                'g_recorder_info': {
                    'model_record_base': self.record_base.storage,
                    'batch_size': self.batch_size,
                    'records': self.records.storage,
                }
            }
        )
        func = func.constants({'k_debug_recording_enabled': 1})
        return func

    def dump_data(self):
        """Append this iteration and write its array payload."""

        json_path = os.path.join(self.out_folder, 'recordings.json')

        if os.path.exists(json_path):
            root = json.loads(open(json_path).read())
        else:
            root = {
                'types': {},
                'recordings': [],
            }

        result_data: dict[str, np.ndarray] = {}

        def check_data(type: dict[str, Any], data: np.ndarray):
            if type.get('children', []):
                return sum(check_data(c, data) for c in type['children'])
            elif type['type'] in ('half', 'float'):
                dtype = np.float32 if type['type'] == 'float' else np.float16
                col = data[..., type['offset'] : type['offset'] + type['size']].view(dtype)[..., 0]
                return int(np.sum(np.isnan(col)) + np.sum(np.isinf(col)))
            return 0

        def recurse_model(m: IModel, name: str | None):
            parameters = []
            params = m.model_params()
            for param in params:
                p_name = f'parameter_{len(result_data)}'
                result_data[p_name] = param.to_numpy()
                d_name = ''
                parameters.append((p_name, d_name))

            records = {}
            model_bad_count = 0
            if m.debuggable:
                for record in m.debug_records:
                    key = (m.model_id, record.index)
                    if key not in self.record_data:
                        continue
                    buffer = self.record_data[key]

                    layout = buffer.dtype.buffer_layout.reflection
                    type_name = buffer.dtype.full_name
                    if type_name not in root['types']:
                        root['types'][type_name] = self.type_to_json(layout, 0) | {
                            'stride': layout.stride
                        }

                    r_name = f'record_{len(result_data):05d}'
                    batch_data = buffer.storage.to_numpy().reshape(self.batch_size, layout.stride)
                    bad_count = check_data(root['types'][type_name], batch_data)
                    model_bad_count += bad_count

                    result_data[r_name] = batch_data
                    records[record.name] = {
                        'type': type_name,
                        'data': r_name,
                        'bad_values': bad_count,
                    }

            def expand_model_chain(x: IModel):
                if isinstance(x, ModelChain):
                    return functools.reduce(
                        operator.iadd, (expand_model_chain(c) for c in x.models), []
                    )
                else:
                    return [x]

            if isinstance(m, ModelChain):
                children = [recurse_model(c, None) for c in expand_model_chain(m)]
            else:
                children = [recurse_model(c, m.child_name(c)) for c in m.children()]

            model_bad_count += sum(c['bad_values'] for c in children)

            result: dict[str, Any] = {'type': type(m).__name__, 'bad_values': model_bad_count}
            if parameters:
                result['parameters'] = parameters
            if records:
                result['records'] = records
            if children:
                result['children'] = children
            if name is not None:
                result['name'] = name

            return result

        models = [recurse_model(m, m.name) for m in self.models]

        data_file = f'recording-{self.iteration:06d}.npz'
        data_path = os.path.join(self.out_folder, data_file)

        root['recordings'].append(
            {'iteration': self.iteration, 'models': models, 'data': data_file}
        )

        open(json_path, 'w').write(json.dumps(root, indent=4))
        np.savez(data_path, **result_data)
