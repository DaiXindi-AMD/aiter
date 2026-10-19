# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import unittest

from aiter.ops.triton.utils import mxfp4_contract as c


class _Storage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class _Tensor:
    def __init__(self, shape, strides, offset=0, size=4096, pointer=4096):
        self.shape = shape
        self._strides = strides
        self.offset = offset
        self.storage = _Storage(pointer, size)
        self.device = "device0"
        self.dtype = "torch.bfloat16"

    def stride(self):
        return self._strides

    def storage_offset(self):
        return self.offset

    def element_size(self):
        return 2

    def untyped_storage(self):
        return self.storage

    def data_ptr(self):
        return self.storage.pointer + self.offset * 2


class TestMxfp4HostContract(unittest.TestCase):
    def test_rng_full_unsigned_limbs(self):
        self.assertEqual(
            c.rng_words(1, (1, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0x80000000, 4, 0), 4),
            (c.U64_MAX, 1 << 63),
        )
        self.assertEqual(
            c.rng_words(1, (1, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0), 0), (0, c.U64_MAX)
        )
        for receipt in (
            (1, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 1, 0),
            (0, 0, 0, 0, 0, 1, 0),
            (1, True, 0, 0, 0, 1, 0),
        ):
            with self.assertRaises(c.Mxfp4ContractError) as caught:
                c.rng_words(1, receipt, 1)
            self.assertEqual(caught.exception.reason, 114)

    def test_rne_cannot_consume_rng(self):
        self.assertEqual(c.rng_words(0, (0,) * 7, 32), (0, 0))
        with self.assertRaises(c.Mxfp4ContractError):
            c.rng_words(0, (1, 0, 0, 0, 0, 0, 0), 32)

    def test_metadata_offsets_and_alias(self):
        a = c.tensor_metadata(_Tensor((4, 8), (8, 1), offset=16))
        b = c.tensor_metadata(_Tensor((4, 8), (8, 1), offset=32))
        self.assertNotEqual(a.pointer, b.pointer)
        with self.assertRaises(c.Mxfp4ContractError) as caught:
            c.disjoint((a, b))
        self.assertEqual(caught.exception.reason, 105)
        c.disjoint((a, c.tensor_metadata(_Tensor((4, 8), (8, 1), offset=48))))

    def test_capacity_strides_and_empty(self):
        for tensor in (
            _Tensor((32, 32), (32, 1), size=16),
            _Tensor((2, 2), (0, 1)),
            _Tensor((2, 2), (-1, 1)),
            _Tensor((2, 2), (1 << 31, 1), size=1 << 40),
        ):
            with self.assertRaises(c.Mxfp4ContractError):
                c.tensor_metadata(tensor)
        empty = c.tensor_metadata(_Tensor((0, 32), (32, 1), size=0, pointer=0))
        self.assertEqual(empty.span, 0)
        c.disjoint((empty, empty))

    def test_shape_and_headers(self):
        s = (1, 0, 950, 256, 1, 4096, 2048, 33, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1)
        self.assertEqual(c.validate_static_spec(s), s)
        shapes = dict(c.buffer_shapes(33, 4096, 2048))
        self.assertEqual(shapes["x_t_out"], (2048, 32))
        self.assertEqual(shapes["dy_t_scales_out"], (4096, 2))
        for index, value in ((0, 2), (1, 1), (8, True), (10, 0), (18, 0)):
            bad = list(s)
            bad[index] = value
            with self.assertRaises(c.Mxfp4ContractError):
                c.validate_static_spec(tuple(bad))
        for op, selector, layout in (
            (0, 1, 2),
            (1, 1, 1),
            (2, 1, 2),
            (3, 0, 1),
            (4, 1, 2),
            (5, 0, 0),
        ):
            c.header((1, 0, op, selector, layout, 0), op)

    def test_ready_slots_cannot_change_meaning(self):
        key = (0, -1, 0, 0, 0, 1, 0, 0, 0, "any")
        config = (10000001, 1, 32, 1, 1, 1, 0)
        r = c.ReadyRecordV1(config, key, "artifact", "code", object(), "device0")
        c.register_ready((r,))
        self.assertIs(c.ready_record(config, 0), r)
        changed = c.ReadyRecordV1(config, key, "different", "code", object(), "device0")
        with self.assertRaises(c.Mxfp4ContractError):
            c.register_ready((changed,))
        with self.assertRaises(c.Mxfp4ContractError):
            c.ready_record((10000002, 1, 32, 1, 1, 1, 0), 0)
