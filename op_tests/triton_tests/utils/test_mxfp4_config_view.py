# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-only projection tests using the checked-in config records, not GPU code."""

import dataclasses
import json
import unittest
from pathlib import Path
from unittest import mock

from aiter.ops.triton.utils import mxfp4_contract as c

PROFILE_ROOT = Path(__file__).parents[3] / "aiter/ops/triton/configs/gfx950/triton"


class TestDenseConfigView(unittest.TestCase):
    def setUp(self):
        self.registry = mock.patch.multiple(c, _READY={}, _COMPANIONS={})
        self.registry.start()
        self.addCleanup(self.registry.stop)
        manifest = json.loads(
            (
                PROFILE_ROOT
                / "gemm/gemm_afp4wfp4/GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json"
            ).read_text()
        )
        self.records = []
        self.expected = {}
        fields = (
            (
                "BLOCK_M",
                "BLOCK_N",
                "NUM_ITER",
                "NUM_STAGES",
                "num_warps",
                "waves_per_eu",
            ),
            ("num_warps", "num_stages", "waves_per_eu"),
            ("num_warps", "num_stages", "waves_per_eu"),
            (
                "BLOCK_M",
                "BLOCK_N_PACKED",
                "num_warps",
                "num_stages",
                "waves_per_eu",
                "reserved_zero",
            ),
            (
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
                "GROUP_SIZE_M",
                "num_warps",
                "num_stages",
                "waves_per_eu",
                "matrix_instr_nonkdim",
                "cache_modifier",
                "NUM_KSPLIT",
            ),
            ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "waves_per_eu"),
        )
        for entry in manifest["records"]:
            op = entry["op_id"]
            ref = entry["profile_ref"]
            directory = (
                PROFILE_ROOT
                / ref["family"]
                / ref["config_name"].lower().replace("-", "_")
            )
            filename = (
                "DEFAULT.json"
                if ref["suffix"] == "DEFAULT"
                else ref["config_name"] + "-" + ref["suffix"] + ".json"
            )
            profile = json.loads((directory / filename).read_text())
            if op == 4:
                values = dict(profile["any"])
                values["cache_modifier"] = {None: 0, "": 0, ".cg": 1, ".ca": 2}[
                    values["cache_modifier"]
                ]
            elif op == 5:
                values = profile["any"]["contract_v1"]
            else:
                values = profile[entry["kernel_symbol"].rsplit(".", 1)[1]][
                    "contract_v1"
                ]
            config = (entry["config_id"],) + tuple(values[name] for name in fields[op])
            key = tuple(
                entry[name]
                for name in (
                    "op_id",
                    "pass_id",
                    "variant",
                    "source_dtype",
                    "output_dtype",
                    "b_scale_row_group",
                    "linear_n",
                    "linear_k",
                    "output_mode",
                )
            ) + ("any",)
            launcher = mock.Mock(
                side_effect=AssertionError("metadata must not execute code")
            )
            record = c.ReadyRecordV1(
                config, key, "fixture-artifact", "fixture-code", launcher, "device0"
            )
            self.records.append(record)
            self.expected[key] = config
        self.table = c.register_ready(self.records)

    def view(self, k=2048, mode=1, dtype=1, capacity=33, table=None):
        spec = (
            1,
            0,
            950,
            256,
            1,
            4096,
            k,
            capacity,
            0,
            mode,
            dtype,
            1,
            2,
            1,
            1,
            1,
            2,
            0,
            1,
        )
        return c._prepared_config_view_v1(
            spec, self.table if table is None else table, "fixture-artifact", "device0"
        )

    def test_real_record_correspondence(self):
        for k in (2048, 4096):
            for mode, dtype in ((0, 0), (0, 1), (1, 1)):
                view = self.view(k, mode, dtype)
                self.assertEqual(len(view.entries), 20)
                for pass_id in range(3):
                    output, group, out_mode = (
                        (dtype, 1, mode) if pass_id == 2 else (0, 32, 0)
                    )
                    expected = self.expected[
                        (4, pass_id, 0, 0, output, group, 4096, k, out_mode, "any")
                    ]
                    entry = view.gemm(pass_id)
                    self.assertEqual(entry.config, expected)
                    self.assertEqual(entry.config_id, expected[0])
                for record in self.records:
                    op, _, variant, source, output, *_ = record.semantic_key
                    if op != 4:
                        entry = view.primitive(
                            op,
                            variant=variant,
                            source_dtype=source,
                            output_dtype=output,
                        )
                        self.assertEqual(entry.config, record.config)

    def test_empty_same_static_config_identity(self):
        empty, normal = self.view(capacity=0), self.view(capacity=257)
        self.assertEqual(empty, normal)
        for p in range(3):
            self.assertEqual(empty.gemm(p), normal.gemm(p))
        for dtype in (0, 1):
            self.assertEqual(
                empty.primitive(5, variant=4, output_dtype=dtype),
                normal.primitive(5, variant=4, output_dtype=dtype),
            )

    def test_immutable_snapshot(self):
        source = dict(self.table)
        view = self.view(table=source)
        before = view.entries
        source.clear()
        self.assertEqual(view.entries, before)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            view.entries = ()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            view.entries[0].config = ()
        with self.assertRaises(TypeError):
            view.entries[0].config[0] = 0
        self.assertFalse(hasattr(view, "__dict__"))
        self.assertFalse(hasattr(view.entries[0], "__dict__"))

    def test_no_private_objects_or_callbacks(self):
        view = self.view()

        def check(value):
            if dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    self.assertNotIn(
                        field.name,
                        ("semantic_key", "executable", "records", "device", "_READY"),
                    )
                    check(getattr(value, field.name))
            elif isinstance(value, tuple):
                for item in value:
                    check(item)
            else:
                self.assertIs(type(value), int)

        check(view)
        with mock.patch("builtins.open", side_effect=AssertionError("lookup IO")):
            self.assertIs(view.gemm(0), view.gemm(0))
            self.assertIs(view.primitive(2, variant=3), view.primitive(2, variant=3))
        for record in self.records:
            record.executable.assert_not_called()

    def test_missing_is_typed_failure(self):
        partial = {
            key: record for key, record in self.table.items() if key[0] not in (4, 5)
        }
        view = self.view(table=partial)
        for call in (
            lambda: view.gemm(0),
            lambda: view.gemm(2),
            lambda: view.primitive(5, variant=4),
        ):
            with self.assertRaises(c.Mxfp4ContractError) as caught:
                call()
            self.assertEqual(
                (
                    caught.exception.phase,
                    caught.exception.reason,
                    caught.exception.disposition,
                ),
                (2, 111, 1),
            )

    def test_invalid_query_does_not_select_alternative(self):
        view = self.view()
        for call in (
            lambda: view.gemm(True),
            lambda: view.gemm(3),
            lambda: view.primitive(4, variant=0),
            lambda: view.primitive(2, variant=0),
            lambda: view.primitive(0, variant=0, output_dtype=1),
        ):
            with self.assertRaises(c.Mxfp4ContractError) as caught:
                call()
            self.assertEqual(caught.exception.reason, 102)

    def test_projection_requires_registered_identity(self):
        record = self.records[0]
        clone = dataclasses.replace(record, executable=object())
        with self.assertRaises(c.Mxfp4ContractError) as caught:
            self.view(table={record.semantic_key: clone})
        self.assertEqual(caught.exception.reason, 106)

    def test_order_is_canonical_not_input_order(self):
        self.assertEqual(
            self.view(table=dict(reversed(tuple(self.table.items())))), self.view()
        )
