# Copyright (c) Huawei Technologies Co., Ltd. 2023-2026. All rights reserved.
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

from __future__ import annotations

import pytest
import torch

from fused_cpp.moe import bf16_tiled


@pytest.mark.parametrize(
    ("text", "expected"),
    (("64K", 64 * 1024), ("2M", 2 * 1024**2), ("32M", 32 * 1024**2), ("1G", 1024**3)),
)
def test_parse_hugetlb_size(text: str, expected: int) -> None:
    assert bf16_tiled._parse_hugetlb_size(text) == expected


def test_copy_packed_tensor_to_file_backed_storage_survives_unlink(tmp_path) -> None:
    source = torch.arange(64, dtype=torch.float32).to(torch.bfloat16).reshape(8, 8)

    target = bf16_tiled._copy_packed_tensor_to_hugetlbfs(
        source,
        mount_path=tmp_path,
        page_size=4096,
        label="test",
    )

    torch.testing.assert_close(target, source)
    assert target.is_contiguous()
    assert list(tmp_path.iterdir()) == []


def test_hugetlbfs_page_size_rejects_regular_directory(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="not a hugetlbfs mount|requires Linux hugetlbfs"):
        bf16_tiled._hugetlbfs_page_size(tmp_path)


def test_hugetlbfs_page_size_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="configured hugetlbfs path does not exist|requires Linux hugetlbfs"):
        bf16_tiled._hugetlbfs_page_size(tmp_path / "missing")


def test_maybe_move_packed_weights_is_identity_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("FUSED_CPP_MOE_HUGETLBFS_PATH", raising=False)
    w13 = torch.empty(8, dtype=torch.bfloat16)
    w2 = torch.empty(4, dtype=torch.bfloat16)

    moved_w13, moved_w2 = bf16_tiled._maybe_move_packed_weights_to_hugetlbfs(w13, w2)

    assert moved_w13 is w13
    assert moved_w2 is w2
