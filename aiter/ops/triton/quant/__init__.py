"""Public Triton quantization APIs with lazy submodule loading."""

from importlib import import_module


_EXPORTS = {
    # fused_fp8_quant.py
    "calc_rows_per_block": ".fused_fp8_quant",
    "fused_flatten_fp8_group_quant": ".fused_fp8_quant",
    "fused_reduce_act_mul_fp8_group_quant": ".fused_fp8_quant",
    "fused_reduce_rms_fp8_group_quant": ".fused_fp8_quant",
    "fused_rms_fp8_group_quant": ".fused_fp8_quant",
    "fused_rms_fp8_per_tensor_static_quant": ".fused_fp8_quant",
    "fused_rms_gated_fp8_group_quant": ".fused_fp8_quant",
    "get_fp8_min_max_bounds": ".fused_fp8_quant",
    # fused_mxfp4_quant.py
    "fused_dynamic_mxfp4_quant_moe_sort": ".fused_mxfp4_quant",
    "fused_flatten_mxfp4_quant": ".fused_mxfp4_quant",
    "fused_reduce_act_mul_and_mxfp4_quant": ".fused_mxfp4_quant",
    "fused_reduce_rms_mxfp4_quant": ".fused_mxfp4_quant",
    "fused_rms_mxfp4_quant": ".fused_mxfp4_quant",
    # fused_mxfp8_quant.py
    "fused_dual_rmsnorm_mxfp8_quant": ".fused_mxfp8_quant",
    "fused_flatten_mxfp8_quant": ".fused_mxfp8_quant",
    "fused_rms_mxfp8_quant": ".fused_mxfp8_quant",
    # mxfp4.py
    "convert_from_mxfp4": ".mxfp4",
    "convert_from_mxfp4_2d": ".mxfp4",
    "convert_to_mxfp4": ".mxfp4",
    "convert_to_mxfp4_2d": ".mxfp4",
    "dequant_hadamard_quant_mxfp4": ".mxfp4",
    "dequant_transpose_mxfp4": ".mxfp4",
    "dual_layout_quant_mxfp4": ".mxfp4",
    "hadamard_quant_mxfp4": ".mxfp4",
    "hadamard_transform": ".mxfp4",
    "mxfp4_data_shuffle_supported": ".mxfp4",
    "mxfp4_scale_swizzle_supported": ".mxfp4",
    "swizzle_expanded_mxfp4_scale": ".mxfp4",
    "swizzle_mxfp4_scale": ".mxfp4",
    "transpose_packed_fp4": ".mxfp4",
    # quant.py
    "_mxfp4_quant_op": ".quant",
    "_mxfp8_quant_op": ".quant",
    "_nvfp4_quant_op": ".quant",
    "dynamic_mxfp4_quant": ".quant",
    "dynamic_mxfp8_quant": ".quant",
    "dynamic_mxfp8_quant_n32k4_mbn": ".quant",
    "dynamic_nvfp4_quant": ".quant",
    "dynamic_per_tensor_quant_fp8_i8": ".quant",
    "dynamic_per_token_quant_fp8_i8": ".quant",
    "fp8_legacy_to_mxfp8": ".quant",
    "static_per_tensor_quant_fp8_i8": ".quant",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
