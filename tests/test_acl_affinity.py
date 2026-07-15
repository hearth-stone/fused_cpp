# -*- coding: utf-8 -*-
"""ACL 核心绑定策略控制单元测试。

测试 set_acl_affinity / get_acl_affinity 的设置与查询一致性，
无效核心范围的处理，以及核心绑定变更后 GEMM 仍能正常执行。
"""

import os
import platform
import sys

import pytest
import torch

_is_aarch64 = platform.machine() in ("aarch64", "arm64")
_is_linux = sys.platform.startswith("linux")

try:
    import fused_cpp

    _acl_available = fused_cpp._supports_acl
except ImportError:
    _acl_available = False

pytestmark = pytest.mark.skipif(
    not (_is_aarch64 and _acl_available),
    reason="ACL 核心绑定测试仅在 AArch64 平台且 ACL 可用时运行",
)


class TestACLAffinitySetGet:
    """测试核心绑定策略的设置与查询。"""

    def test_initial_state(self) -> None:
        """初始状态应返回 (-1, -1, -1) 或上次设置的值。"""
        result = fused_cpp.get_acl_affinity()
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.skipif(not _is_linux, reason="核心绑定仅在 Linux 上可用")
    def test_set_and_get_consistency(self) -> None:
        """设置后查询应返回一致的值。"""
        num_cpus = os.cpu_count() or 1
        # 使用安全的核心范围
        core_start = 0
        core_end = min(2, num_cpus)
        num_threads = core_end - core_start

        fused_cpp.set_acl_affinity(core_start, core_end, num_threads)
        result = fused_cpp.get_acl_affinity()

        assert result == (core_start, core_end, num_threads)

    def test_invalid_range_start_ge_end(self) -> None:
        """core_start >= core_end 应被忽略，不改变当前配置。"""
        # 先记录当前状态
        before = fused_cpp.get_acl_affinity()

        # 设置无效范围（start >= end）
        fused_cpp.set_acl_affinity(5, 3, 2)

        after = fused_cpp.get_acl_affinity()
        assert before == after

    @pytest.mark.skipif(not _is_linux, reason="核心绑定仅在 Linux 上可用")
    def test_invalid_range_exceeds_cpu_count(self) -> None:
        """core_end 超出系统核心数应被忽略。"""
        before = fused_cpp.get_acl_affinity()

        num_cpus = os.cpu_count() or 1
        # 设置超出系统核心数的范围
        fused_cpp.set_acl_affinity(0, num_cpus + 100, 4)

        after = fused_cpp.get_acl_affinity()
        assert before == after


class TestACLAffinityWithGEMM:
    """测试核心绑定变更后 GEMM 仍能正常执行。"""

    @pytest.mark.skipif(not _is_linux, reason="核心绑定仅在 Linux 上可用")
    def test_gemm_after_affinity_change(self) -> None:
        """修改核心绑定后，已创建的 handler 仍应能正常执行 GEMM。"""
        K, N = 128, 256
        weight = torch.randn(K, N, dtype=torch.float32)
        handler = fused_cpp.create_acl_gemm(weight)

        # 先执行一次 GEMM
        x = torch.randn(4, K, dtype=torch.float32)
        result_before = fused_cpp.acl_gemm(handler, x, None)

        # 修改核心绑定
        num_cpus = os.cpu_count() or 1
        core_end = min(2, num_cpus)
        fused_cpp.set_acl_affinity(0, core_end, core_end)

        # 再次执行 GEMM，应仍然正常
        result_after = fused_cpp.acl_gemm(handler, x, None)

        # 两次结果应一致（相同输入、相同权重）
        torch.testing.assert_close(result_before, result_after, atol=1e-5, rtol=1e-5)
