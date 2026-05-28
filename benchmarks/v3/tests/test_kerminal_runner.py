from __future__ import annotations

import time
from pathlib import Path

from src.eval import benchmark, kerminal_runner
from src.eval.context import augment_system_prompt, build_initial_user_message
from src.prompts import get_system_prompt


class FakeSandbox:
    def __init__(self, *, has_solution: bool, results: list[dict] | None = None):
        self._has_solution = has_solution
        self.results = list(results or [])
        self.commands: list[tuple[str, int | None]] = []
        self.workspace_path = Path("/tmp/kb_fake_workspace")

    def file_exists(self, path: str) -> bool:
        return path == "solution.py" and self._has_solution

    def run_command(self, command: str, timeout: int | None = None) -> dict:
        self.commands.append((command, timeout))
        if not self.results:
            raise AssertionError(f"unexpected command: {command}")
        return self.results.pop(0)


class FakeClient:
    seen_cwd: Path | None = None

    @classmethod
    async def start(cls, options):
        cls.seen_cwd = Path.cwd()
        return object()


def test_start_kerminal_client_in_workspace_changes_and_restores_cwd(tmp_path):
    previous_cwd = Path.cwd()
    result = kerminal_runner.asyncio.run(
        kerminal_runner._start_kerminal_client_in_workspace(
            FakeClient,
            object(),
            tmp_path,
        )
    )
    assert result is not None
    assert FakeClient.seen_cwd == tmp_path
    assert Path.cwd() == previous_cwd


def test_map_prompt_workspace_rewrites_modal_workspace_paths():
    mapped = kerminal_runner._map_prompt_workspace(
        "Write /workspace/solution.py and run cd /workspace && python test.py",
        Path("/tmp/kernelbench_local_abc"),
    )
    assert "/workspace" not in mapped
    assert "/tmp/kernelbench_local_abc/solution.py" in mapped
    assert "cd /tmp/kernelbench_local_abc && python test.py" in mapped


def test_self_check_passed_requires_standalone_pass_line():
    assert kerminal_runner._self_check_passed("max_diff=0.0\nPASS\n")
    assert not kerminal_runner._self_check_passed("max_diff=1.0\nFAIL\n")
    assert not kerminal_runner._self_check_passed("NOT PASS\n")


def test_feedback_truncation_keeps_head_and_tail():
    text = "a" * 20 + "b" * 20 + "c" * 20
    truncated = kerminal_runner._truncate_for_feedback(text, limit=20)
    assert truncated.startswith("a" * 10)
    assert "truncated 40 chars" in truncated
    assert truncated.endswith("c" * 10)


def test_missing_solution_feedback_points_to_current_workspace():
    feedback = kerminal_runner._missing_solution_feedback(Path("/tmp/run_123"))
    assert "solution.py" in feedback
    assert "/tmp/run_123" in feedback
    assert "/workspace/solution.py" not in feedback


def test_network_retry_budget_defaults_to_two(monkeypatch):
    monkeypatch.delenv("KERMINAL_NETWORK_RETRIES", raising=False)
    assert kerminal_runner._network_retry_budget() == 2


def test_network_retry_budget_accepts_env_override(monkeypatch):
    monkeypatch.setenv("KERMINAL_NETWORK_RETRIES", "4")
    assert kerminal_runner._network_retry_budget() == 4


def test_network_retry_budget_invalid_env_uses_default(monkeypatch):
    monkeypatch.setenv("KERMINAL_NETWORK_RETRIES", "invalid")
    assert kerminal_runner._network_retry_budget() == 2


def test_is_retryable_kerminal_error_matches_network_failures():
    assert kerminal_runner._is_retryable_kerminal_error("stream_error: 503 service unavailable")
    assert kerminal_runner._is_retryable_kerminal_error("connection reset by peer")
    assert kerminal_runner._is_retryable_kerminal_error("kerminal process exited: code=1")
    assert not kerminal_runner._is_retryable_kerminal_error("max_turns_exceeded")
    assert not kerminal_runner._is_retryable_kerminal_error("timeout_exceeded")
    assert not kerminal_runner._is_retryable_kerminal_error("failed to import or compile")




def test_internal_stream_retry_notice_is_not_fatal():
    assert kerminal_runner._is_internal_stream_retry_message("Retrying... 1/4")
    assert kerminal_runner._is_internal_stream_retry_message("retrying request 2/4")
    assert not kerminal_runner._is_internal_stream_retry_message("503 service unavailable")
    assert not kerminal_runner._is_internal_stream_retry_message("Retry attempts exhausted")


def test_with_accumulated_usage_preserves_result_status():
    result = kerminal_runner.KerminalRunResult(
        submitted=True,
        solution_path="solution.py",
        turns_used=1,
        input_tokens=10,
        output_tokens=20,
        cache_creation_tokens=3,
        cache_read_tokens=4,
        error=None,
    )
    accumulated = kerminal_runner._with_accumulated_usage(
        result,
        turns_used=3,
        input_tokens=30,
        output_tokens=40,
        cache_creation_tokens=5,
        cache_read_tokens=6,
    )
    assert accumulated.submitted
    assert accumulated.solution_path == "solution.py"
    assert accumulated.turns_used == 3
    assert accumulated.input_tokens == 30
    assert accumulated.output_tokens == 40
    assert accumulated.cache_creation_tokens == 5
    assert accumulated.cache_read_tokens == 6


def test_verify_solution_reports_missing_file_without_running_commands():
    sandbox = FakeSandbox(has_solution=False)
    result = kerminal_runner._verify_solution(
        sandbox=sandbox,
        is_metal=False,
        turn=1,
        deadline=time.monotonic() + 30,
        artifact_dir=None,
    )
    assert not result.submitted
    assert result.status == "missing_solution"
    assert result.feedback and "No solution.py" in result.feedback
    assert sandbox.commands == []


def test_verify_solution_returns_compile_feedback_on_import_failure():
    sandbox = FakeSandbox(
        has_solution=True,
        results=[
            {"returncode": 1, "stdout": "", "stderr": "boom", "timed_out": False},
            {"returncode": 1, "stdout": "", "stderr": "still boom", "timed_out": False},
        ],
    )
    result = kerminal_runner._verify_solution(
        sandbox=sandbox,
        is_metal=False,
        turn=1,
        deadline=time.monotonic() + 30,
        artifact_dir=None,
    )
    assert not result.submitted
    assert result.status == "compile_failed"
    assert result.feedback and "failed to import or compile" in result.feedback
    assert "boom" in result.feedback
    assert len(sandbox.commands) == 2


def test_verify_solution_submits_after_self_check_pass():
    sandbox = FakeSandbox(
        has_solution=True,
        results=[
            {"returncode": 0, "stdout": "OK\n", "stderr": "", "timed_out": False},
            {"returncode": 0, "stdout": "max_diff=0.000000\nPASS\n", "stderr": "", "timed_out": False},
        ],
    )
    result = kerminal_runner._verify_solution(
        sandbox=sandbox,
        is_metal=False,
        turn=1,
        deadline=time.monotonic() + 30,
        artifact_dir=None,
    )
    assert result.submitted
    assert result.solution_path == "solution.py"
    assert result.status == "self_check_passed"
    assert len(sandbox.commands) == 2


def test_finalize_preserves_existing_error_when_no_solution_submitted():
    from src.eval.agent import _finalize_agent_result
    from src.eval.results import EvalResult
    from src.models import ModelConfig

    class FakeHardwareTarget:
        is_metal = False
        gpu_sku = "A100"

    sandbox = FakeSandbox(
        has_solution=False,
        results=[
            {"returncode": 1, "stdout": "", "stderr": "", "timed_out": False},
            {"returncode": 1, "stdout": "", "stderr": "", "timed_out": False},
        ],
    )
    result = EvalResult(error="stream_error: Retrying... 1/4")

    _finalize_agent_result(
        result=result,
        submitted=False,
        solution_path=None,
        input_tokens=0,
        output_tokens=0,
        turns_used=1,
        model_config=ModelConfig(name="Kerminal Default", model_id="default", provider="kerminal"),
        sandbox=sandbox,
        hardware_target=FakeHardwareTarget(),
        level=1,
    )

    assert result.error == "stream_error: Retrying... 1/4"


def test_candidate_selection_uses_highest_correct_speedup_and_keeps_ties():
    incorrect_fast = kerminal_runner._CandidateResult(
        turn=1,
        solution_path="turn_1_solution.py",
        benchmark_result={"correct": False, "speedup": 100.0},
    )
    first = kerminal_runner._CandidateResult(
        turn=2,
        solution_path="turn_2_solution.py",
        benchmark_result={"correct": True, "speedup": 1.5},
    )
    better = kerminal_runner._CandidateResult(
        turn=3,
        solution_path="turn_3_solution.py",
        benchmark_result={"correct": True, "speedup": 2.0},
    )
    tie = kerminal_runner._CandidateResult(
        turn=4,
        solution_path="turn_4_solution.py",
        benchmark_result={"correct": True, "speedup": 2.0},
    )

    assert not kerminal_runner._candidate_is_better(incorrect_fast, None)
    assert kerminal_runner._candidate_is_better(first, None)
    assert kerminal_runner._candidate_is_better(better, first)
    assert not kerminal_runner._candidate_is_better(tie, better)


def test_benchmark_solution_path_maps_workspace_paths_to_relative():
    assert benchmark._benchmark_solution_path("/workspace/turn_1_solution.py") == "turn_1_solution.py"
    assert benchmark._benchmark_solution_path("solution.py") == "solution.py"


def test_best_of_turns_prompt_removes_submit_immediately_instruction():
    prompt = augment_system_prompt("base", is_metal=False, best_of_turns=True)
    assert "submit immediately" not in prompt
    assert "harness will snapshot" in prompt
    assert "Do NOT present a candidate until PASS" in prompt


def test_regular_prompt_keeps_submit_instruction():
    prompt = augment_system_prompt("base", is_metal=False, best_of_turns=False)
    assert "If PASS: submit immediately" in prompt


def test_best_of_turns_initial_message_describes_harness_feedback():
    msg = build_initial_user_message(
        hardware_name="a100_local",
        problem_name="1_Test.py",
        level=1,
        gpu_name="A100",
        max_turns=3,
        reference_code="class Model: pass",
        metadata={"op_type": "gemm", "supported_precisions": ["fp32"]},
        best_of_turns=True,
    )
    assert "Use up to 3 optimization turns" in msg
    assert "harness will benchmark" in msg
    assert "Do not try to import or call a submit module" in msg


def test_kerminal_system_prompt_omits_submit_tool_and_instruction():
    prompt = get_system_prompt(
        hardware_name="a100_local",
        gpu_name="A100",
        vram_gb=80,
        include_submit_tool=False,
    )
    assert "submit(solution_path)" not in prompt
    assert "Submit ONLY after PASS" not in prompt
    assert "harness will benchmark it" in prompt
