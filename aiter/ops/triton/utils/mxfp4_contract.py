# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Checked host metadata for the dense MXFP4 v1 contract.

This module has no device, file, compiler, or tensor allocation side effects.
"""

import functools
import inspect
from dataclasses import dataclass
from enum import IntEnum
from threading import RLock
from types import MappingProxyType

U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
SPAN_MAX = (1 << 31) - 1


class Phase(IntEnum):
    QUERY = 0
    PREPARE = 1
    BIND = 2
    ATTACH_POSTCOMMIT = 3
    LAUNCH = 4
    POSTLAUNCH_HEALTH = 5
    ASYNC_DEVICE = 6


class Fatal(IntEnum):
    ABI_VERSION_MISMATCH = 100
    NUMERIC_LAYOUT_MISMATCH = 101
    INVALID_PUBLIC_ARGUMENT = 102
    CAPACITY_OR_INDEX_OVERFLOW = 103
    WORKSPACE_SLICE_MISMATCH = 104
    ILLEGAL_ALIAS = 105
    PREPARE_FAILURE = 106
    PREPARE_RESOURCE_UNSUPPORTED = 107
    ACTUAL_OOM = 108
    RUNTIME_OPERAND_CONTRACT_VIOLATION = 109
    BOUND_KERNEL_FAILURE = 110
    UNPREPARED_SPECIALIZATION = 111
    NUMERIC_NONFINITE = 112
    ASYNCHRONOUS_DEVICE_ERROR = 113
    SR_TOKEN_MISMATCH = 114


class OutputMode(IntEnum):
    OVERWRITE = 0
    ACCUMULATE_FP32 = 1


class Mxfp4ContractError(RuntimeError):
    """A typed fatal error; messages must not select a fallback."""

    def __init__(self, phase, reason, op_id=-1, message=""):
        self.phase = int(phase)
        self.reason = int(reason)
        self.disposition = 1
        self.op_id = op_id
        super().__init__(message or Fatal(reason).name)


def fail(phase, reason=Fatal.INVALID_PUBLIC_ARGUMENT, op_id=-1, message=""):
    raise Mxfp4ContractError(phase, reason, op_id, message)


def integer(value, low=0, high=(1 << 63) - 1, *, phase=Phase.BIND, reason=102):
    """Check wire integers without coercion, including rejection of bool."""
    if type(value) is not int or not low <= value <= high:
        fail(phase, reason)
    return value


def flat_tuple(value, size, *, phase=Phase.BIND, reason=102):
    if type(value) is not tuple or len(value) != size:
        fail(phase, reason)
    for item in value:
        integer(item, phase=phase, reason=reason)
    return value


def pad32(rows):
    integer(rows, high=U64_MAX)
    result = (rows + 31) // 32 * 32
    if result > U64_MAX:
        fail(Phase.BIND, Fatal.CAPACITY_OR_INDEX_OVERFLOW)
    return result


def rng_words(rounding, receipt, count, *, phase=Phase.BIND):
    """Validate the seven wire fields and return full unsigned seed/base."""
    integer(rounding, high=1, phase=phase, reason=114)
    flat_tuple(receipt, 7, phase=phase, reason=114)
    integer(count, high=U64_MAX, phase=phase, reason=114)
    if rounding == 0:
        if receipt != (0,) * 7:
            fail(phase, Fatal.SR_TOKEN_MISMATCH)
        return 0, 0
    if receipt[0] != 1:
        fail(phase, Fatal.SR_TOKEN_MISMATCH)
    for limb in receipt[1:]:
        integer(limb, high=U32_MAX, phase=phase, reason=114)
    seed = receipt[1] | receipt[2] << 32
    base = receipt[3] | receipt[4] << 32
    reserved = receipt[5] | receipt[6] << 32
    if reserved != count or base + reserved > U64_MAX:
        fail(phase, Fatal.SR_TOKEN_MISMATCH)
    return seed, base


def header(value, op_id, *, phase=Phase.BIND):
    flat_tuple(value, 6, phase=phase)
    if value[:2] != (1, 0):
        fail(phase, Fatal.ABI_VERSION_MISMATCH, op_id)
    selector = 0 if op_id in (3, 5) else 1
    physical = 0 if op_id == 5 else (1 if op_id in (1, 3) else 2)
    if value[2:] != (op_id, selector, physical, 0):
        fail(phase, Fatal.NUMERIC_LAYOUT_MISMATCH, op_id)


@dataclass(frozen=True)
class TensorMetadataV1:
    shape: tuple
    strides: tuple
    dtype: str
    device: object
    pointer: int
    storage_pointer: int
    storage_offset: int
    storage_bytes: int
    element_bytes: int
    span: int

    @property
    def byte_range(self):
        return self.pointer, self.pointer + self.span


def tensor_metadata(tensor, *, phase=Phase.BIND):
    """Read tensor metadata only; never read a device value or make a view."""
    try:
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
        size = tensor.element_size()
        offset = tensor.storage_offset()
        storage = tensor.untyped_storage()
        storage_pointer = storage.data_ptr()
        storage_bytes = storage.nbytes()
        pointer = tensor.data_ptr()
        dtype = str(tensor.dtype)
        device = tensor.device
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise Mxfp4ContractError(phase, 102, message="invalid tensor metadata") from exc
    integer(size, 1, 8, phase=phase)
    integer(offset, high=U64_MAX, phase=phase)
    integer(pointer, high=U64_MAX, phase=phase)
    integer(storage_pointer, high=U64_MAX, phase=phase)
    integer(storage_bytes, high=U64_MAX, phase=phase)
    if len(shape) != len(strides):
        fail(phase)
    for dimension in shape:
        integer(dimension, high=U64_MAX, phase=phase)
    empty = 0 in shape
    for stride in strides:
        integer(stride, 0 if empty else 1, U64_MAX, phase=phase)
    span = 0 if empty else (sum((s - 1) * t for s, t in zip(shape, strides)) + 1) * size
    if span > SPAN_MAX or offset * size + span > storage_bytes:
        fail(phase, Fatal.CAPACITY_OR_INDEX_OVERFLOW)
    if storage_pointer + storage_bytes > U64_MAX or pointer + span > U64_MAX:
        fail(phase, Fatal.CAPACITY_OR_INDEX_OVERFLOW)
    if span and (pointer == 0 or pointer != storage_pointer + offset * size):
        fail(phase, Fatal.CAPACITY_OR_INDEX_OVERFLOW)
    if pointer % size:
        fail(phase)
    return TensorMetadataV1(
        shape,
        strides,
        dtype,
        device,
        pointer,
        storage_pointer,
        offset,
        storage_bytes,
        size,
        span,
    )


def check_tensor(
    tensor, shape, dtype, device, *, transpose_scale=False, phase=Phase.BIND
):
    meta = tensor_metadata(tensor, phase=phase)
    if meta.shape != tuple(shape) or meta.dtype != dtype or meta.device != device:
        fail(phase)
    if meta.span:
        canonical = []
        stride = 1
        for dim in reversed(shape):
            canonical.insert(0, stride)
            stride *= dim
        allowed = {tuple(canonical)}
        if transpose_scale and len(shape) == 2:
            allowed.add((1, shape[0]))
        if meta.strides not in allowed:
            fail(phase)
    return meta


def disjoint(writable, readonly=(), *, phase=Phase.BIND):
    """Reject overlaps by reachable bytes, not just tensor start addresses."""
    for index, left in enumerate(writable):
        for right in tuple(writable[index + 1 :]) + tuple(readonly):
            if left.device != right.device or not left.span or not right.span:
                continue
            a, b = left.byte_range
            c, d = right.byte_range
            if max(a, c) < min(b, d):
                fail(phase, Fatal.ILLEGAL_ALIAS)


@dataclass(frozen=True)
class DenseCapabilityV1:
    capability_status: int
    miss_reason: int
    abi_supported_min_minor: int
    abi_supported_max_minor: int
    required_buffer_shapes: tuple
    alignment: tuple
    span_limit: int
    workspace_formula: tuple
    capture_safe: bool
    requires_prepare_resource_check: bool


def buffer_shapes(rows, n, k):
    p = pad32(rows)
    return (
        ("w_fp4", (n, k // 2)),
        ("w_scales", (n // 32, k // 32)),
        ("wt_fp4", (k, n // 2)),
        ("wt_scales", (k // 32, n // 32)),
        ("x_row_out", (rows, k // 2)),
        ("x_row_scales_out", (rows, k // 32)),
        ("x_t_out", (k, p // 2)),
        ("x_t_scales_out", (k, p // 32)),
        ("dy_row_out", (rows, n // 2)),
        ("dy_row_scales_out", (rows, n // 32)),
        ("dy_t_out", (n, p // 2)),
        ("dy_t_scales_out", (n, p // 32)),
        ("y_out", (rows, n)),
        ("dx_out", (rows, k)),
        ("w_out", (n, k)),
        ("workspace", (0,)),
        ("status", (1,)),
    )


def validate_static_spec(static_spec):
    """Check the nineteen signed wire fields without device discovery."""
    s = flat_tuple(static_spec, 19, phase=Phase.QUERY)
    if s[:2] != (1, 0):
        fail(Phase.QUERY, Fatal.ABI_VERSION_MISMATCH)
    if s[4] not in (1, 2, 3) or s[8] != 0 or s[9] not in (0, 1) or s[10] not in (0, 1):
        fail(Phase.QUERY)
    if s[9] == 1 and s[10] != 1:
        fail(Phase.QUERY)
    if s[11:] != (1, 2, 1, 1, 1, 2, 0, 1):
        fail(Phase.QUERY, Fatal.NUMERIC_LAYOUT_MISMATCH)
    if s[3] == 0 or s[5] == 0 or s[6] == 0:
        fail(Phase.QUERY)
    return s


@dataclass(frozen=True)
class ReadyRecordV1:
    config: tuple
    semantic_key: tuple
    artifact_digest: str
    code_digest: str
    executable: object
    device: object


_READY_LOCK = RLock()
_READY = {}
_COMPANIONS = {}


def register_ready(records):
    """Atomically publish code-only slots; a config ID can never change meaning."""
    with _READY_LOCK:
        for record in records:
            prior = _READY.get(record.config[0])
            if prior is not None and (
                prior.config != record.config
                or prior.semantic_key != record.semantic_key
                or prior.artifact_digest != record.artifact_digest
                or prior.code_digest != record.code_digest
                or prior.device != record.device
            ):
                fail(Phase.PREPARE, Fatal.PREPARE_FAILURE)
        for record in records:
            if record.config[0] not in _READY:
                _READY[record.config[0]] = record
            _COMPANIONS[
                (record.artifact_digest, record.device, record.semantic_key)
            ] = _READY[record.config[0]]
    return MappingProxyType(
        {record.semantic_key: _READY[record.config[0]] for record in records}
    )


def companion_record(record, semantic_key):
    """Resolve an exact code-only child from the same prepared artifact."""
    with _READY_LOCK:
        child = _COMPANIONS.get((record.artifact_digest, record.device, semantic_key))
    if child is None:
        fail(Phase.LAUNCH, Fatal.UNPREPARED_SPECIALIZATION)
    return child


def ready_record(config, op_id, *, phase=Phase.LAUNCH):
    lengths = (7, 4, 4, 7, 11, 6)
    flat_tuple(config, lengths[op_id], phase=phase)
    with _READY_LOCK:
        record = _READY.get(config[0])
    if record is None:
        fail(phase, Fatal.UNPREPARED_SPECIALIZATION, op_id)
    if record.config != config or record.semantic_key[0] != op_id:
        fail(phase, Fatal.BOUND_KERNEL_FAILURE, op_id)
    return record


def primitive_source(x, rows, cols, source_dtype, *, phase=Phase.LAUNCH):
    integer(rows, high=SPAN_MAX, phase=phase)
    integer(cols, 32, SPAN_MAX, phase=phase)
    integer(source_dtype, high=1, phase=phase)
    if cols % 32:
        fail(phase)
    dtype = ("torch.bfloat16", "torch.float32")[source_dtype]
    return check_tensor(x, (rows, cols), dtype, x.device, phase=phase)


def status_metadata(status, device, *, phase=Phase.LAUNCH):
    return check_tensor(status, (1,), "torch.int32", device, phase=phase)


_OP_LIBRARIES = []


def prepare_errors(function):
    """Keep selected preparation failures fatal, without candidate probing."""

    @functools.wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Mxfp4ContractError:
            raise
        except MemoryError as exc:
            raise Mxfp4ContractError(Phase.PREPARE, Fatal.ACTUAL_OOM) from exc
        except Exception as exc:
            reason = (
                Fatal.ACTUAL_OOM
                if type(exc).__name__ == "OutOfMemoryError"
                else Fatal.PREPARE_FAILURE
            )
            raise Mxfp4ContractError(Phase.PREPARE, reason) from exc

    return call


def register_contract_op(tensors, mutated, tuples):
    """Register explicit mutation and fixed tuple schemas with a no-output fake."""

    def decorate(function):
        import torch

        arguments = []
        for name in inspect.signature(function).parameters:
            if name in tensors:
                schema_type = "Tensor(" + name + "!)" if name in mutated else "Tensor"
            elif name in tuples:
                schema_type = "(" + ", ".join(["int"] * tuples[name]) + ")"
            else:
                schema_type = "int"
            arguments.append(schema_type + " " + name)
        library = torch.library.Library("aiter", "FRAGMENT")
        library.define(function.__name__ + "(" + ", ".join(arguments) + ") -> ()")
        library.impl(function.__name__, function, "CUDA")

        def fake(*args, **kwargs):
            return None

        library._register_fake(function.__name__, fake)
        _OP_LIBRARIES.append(library)
        operator = getattr(torch.ops.aiter, function.__name__)

        @functools.wraps(function)
        def call(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            for name, value in bound.arguments.items():
                if name in tuples:
                    flat_tuple(value, tuples[name], phase=Phase.LAUNCH)
                elif name not in tensors:
                    integer(value, phase=Phase.LAUNCH)
            return operator(*args, **kwargs)

        return call

    return decorate


def check_record(record, device, source_dtype, variant):
    if record.device != device or record.semantic_key[2:4] != (variant, source_dtype):
        fail(Phase.LAUNCH, Fatal.UNPREPARED_SPECIALIZATION)


@dataclass(frozen=True, slots=True)
class DensePreparedConfigV1:
    """A prepared config's wire metadata, with no executable or tensor reference."""

    op_id: int
    pass_id: int
    variant: int
    source_dtype: int
    output_dtype: int
    b_scale_row_group: int
    output_mode: int
    config: tuple

    @property
    def config_id(self):
        return self.config[0]


@dataclass(frozen=True, slots=True)
class DenseConfigViewV1:
    """Read-only config projection for one prepared dense-training shape."""

    linear_n: int
    linear_k: int
    w_output_mode: int
    w_output_dtype: int
    entries: tuple

    def primitive(self, op_id, *, variant, source_dtype=0, output_dtype=0):
        """Get a prepared row/weight/dual/transpose/zero config, without launching."""
        integer(op_id, high=5)
        integer(variant, high=6)
        integer(source_dtype, high=1)
        integer(output_dtype, high=1)
        variants = ((0, 1), (0,), (2, 3, 5, 6), (0,), (), (4,))
        if variant not in variants[op_id]:
            fail(Phase.BIND, Fatal.INVALID_PUBLIC_ARGUMENT, op_id)
        if (op_id != 5 and output_dtype != 0) or (
            op_id in (3, 5) and source_dtype != 0
        ):
            fail(Phase.BIND, Fatal.INVALID_PUBLIC_ARGUMENT, op_id)
        for entry in self.entries:
            if (entry.op_id, entry.variant, entry.source_dtype, entry.output_dtype) == (
                op_id,
                variant,
                source_dtype,
                output_dtype,
            ):
                return entry
        fail(Phase.BIND, Fatal.UNPREPARED_SPECIALIZATION, op_id)

    def gemm(self, pass_id):
        """Get this bundle's compact F/D or selected W config; empty has no override."""
        integer(pass_id, high=2)
        for entry in self.entries:
            if entry.op_id == 4 and entry.pass_id == pass_id:
                return entry
        fail(Phase.BIND, Fatal.UNPREPARED_SPECIALIZATION, 4)


def _prepared_config_view_v1(static_spec, records, artifact_digest, device):
    """Project registered prepare records only; never retain the records themselves."""
    spec = validate_static_spec(static_spec)
    n, k, mode, dtype = spec[5], spec[6], spec[9], spec[10]
    entries = []
    seen = set()
    for record in records.values():
        if (
            ready_record(record.config, record.semantic_key[0], phase=Phase.PREPARE)
            is not record
            or record.artifact_digest != artifact_digest
            or record.device != device
        ):
            fail(Phase.PREPARE, Fatal.PREPARE_FAILURE)
        op, pass_id, variant, source, output, group, rn, rk, output_mode, bucket = (
            record.semantic_key
        )
        if bucket != "any":
            fail(Phase.PREPARE, Fatal.PREPARE_FAILURE)
        if op == 4:
            if (rn, rk) != (n, k):
                continue
            expected = (dtype, 1, mode) if pass_id == 2 else (0, 32, 0)
            if (output, group, output_mode) != expected:
                continue
        identity = (op, pass_id, variant, source, output, group, output_mode)
        if identity in seen:
            fail(Phase.PREPARE, Fatal.PREPARE_FAILURE)
        seen.add(identity)
        # Copy scalar fields only. The tuple holds already checked wire integers.
        entries.append(DensePreparedConfigV1(*identity, tuple(record.config)))
    entries.sort(
        key=lambda entry: (
            entry.op_id,
            entry.pass_id,
            entry.variant,
            entry.source_dtype,
            entry.output_dtype,
        )
    )
    return DenseConfigViewV1(n, k, mode, dtype, tuple(entries))
