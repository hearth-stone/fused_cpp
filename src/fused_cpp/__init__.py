# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native CPU operators used by vLLM 0.28.0 DeepSeek V4 on Arm."""

from . import bf16_linear, deepseek_v4_attn_gemm_fused, i8gemm, moe
from .bf16_linear import (
    PreparedBF16LinearWeight,
    _supports_bf16_linear,
    linear as bf16_linear_mm,
    linear_raw as bf16_linear_raw,
    prepare as prepare_bf16_linear_weight,
)

__all__ = [
    "PreparedBF16LinearWeight",
    "_supports_bf16_linear",
    "bf16_linear",
    "bf16_linear_mm",
    "bf16_linear_raw",
    "deepseek_v4_attn_gemm_fused",
    "i8gemm",
    "moe",
    "prepare_bf16_linear_weight",
]
