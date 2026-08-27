from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizations.fused_moe_sve.paper_experiments.run_matrix import (
    build_remote_command,
    load_machine,
    load_suite,
    parse_prefixed_json,
    render_argv,
    _tree_digest,
    validate_suite,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "optimizations" / "fused_moe_sve" / "paper_experiments"


@pytest.mark.parametrize("name", ["arm_codex_internal.json", "amazon_ecs_8cores.json"])
def test_machine_configs_are_valid(name: str) -> None:
    machine = load_machine(EXPERIMENT_ROOT / "machines" / name)
    assert machine["project_root"].startswith("/")
    assert machine["affinity_argv"]


@pytest.mark.parametrize("name", ["fused_expert_smoke.json", "fused_expert_pilot.json"])
def test_suite_configs_render_for_every_machine(name: str) -> None:
    suite = load_suite(EXPERIMENT_ROOT / "suites" / name)
    for machine_name in ("arm_codex_internal.json", "amazon_ecs_8cores.json"):
        machine = load_machine(EXPERIMENT_ROOT / "machines" / machine_name)
        context = {
            "machine_id": machine["id"],
            "project_root": machine["project_root"],
            "python": machine["python"],
            "remote_output": "/tmp/result.json",
            **machine["variables"],
        }
        for entry in [*suite["setup"], *suite["cases"]]:
            rendered = render_argv(entry["argv"], context)
            assert rendered
            assert not any("{" in argument or "}" in argument for argument in rendered)


def test_remote_command_quotes_paths_and_environment() -> None:
    command = build_remote_command(
        project_root="/tmp/project with space",
        argv=["python", "script.py", "value with space"],
        environment={"PYTHONPATH": "src"},
        affinity_argv=["taskset", "-c", "0-7"],
    )
    assert command == (
        "cd '/tmp/project with space' && env PYTHONPATH=src taskset -c 0-7 python script.py 'value with space'"
    )


def test_prefixed_json_parser_preserves_all_records() -> None:
    stdout = 'noise\nRESULT_JSON {"variant":"a","median_ms":1.0}\nRESULT_JSON {"variant":"b","median_ms":2.0}\n'
    assert parse_prefixed_json(stdout, "RESULT_JSON ") == [
        {"variant": "a", "median_ms": 1.0},
        {"variant": "b", "median_ms": 2.0},
    ]


def test_output_json_requires_remote_output_placeholder() -> None:
    payload = {
        "schema_version": 1,
        "id": "bad_suite",
        "setup": [],
        "cases": [{"id": "case", "output_json": True, "argv": ["python", "bench.py"]}],
    }
    with pytest.raises(ValueError, match="remote_output"):
        validate_suite(json.loads(json.dumps(payload)))


def test_tree_digest_covers_relative_names_and_contents(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "source.c").write_text("one\n", encoding="utf-8")
    (second / "source.c").write_text("one\n", encoding="utf-8")
    assert _tree_digest(first) == _tree_digest(second)
    (second / "source.c").write_text("two\n", encoding="utf-8")
    assert _tree_digest(first) != _tree_digest(second)
