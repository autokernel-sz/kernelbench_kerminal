"""Kerminal app-server runner for KernelBench v3."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.eval.context import self_check_command


KERMINAL_SDK_SRC = Path("/workspace/kerminal/sdk/python/src")
if KERMINAL_SDK_SRC.exists():
    sdk_path = str(KERMINAL_SDK_SRC)
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

DEFAULT_KERMINAL_NETWORK_RETRIES = 2


@dataclass
class KerminalRunResult:
    submitted: bool
    solution_path: Optional[str]
    turns_used: int
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    error: Optional[str] = None


@dataclass
class _VerificationResult:
    submitted: bool
    solution_path: Optional[str]
    status: str
    feedback: Optional[str] = None
    error: Optional[str] = None


def run_kerminal_agent(
    *,
    model_id: str,
    sandbox,
    system_prompt: str,
    initial_user_message: str,
    max_turns: int,
    max_time: Optional[int],
    is_metal: bool = False,
) -> KerminalRunResult:
    return asyncio.run(
        _run_kerminal_agent_async(
            model_id=model_id,
            sandbox=sandbox,
            system_prompt=system_prompt,
            initial_user_message=initial_user_message,
            max_turns=max_turns,
            max_time=max_time,
            is_metal=is_metal,
        )
    )


async def _run_kerminal_agent_async(
    *,
    model_id: str,
    sandbox,
    system_prompt: str,
    initial_user_message: str,
    max_turns: int,
    max_time: Optional[int],
    is_metal: bool,
) -> KerminalRunResult:
    retry_budget = _network_retry_budget()
    max_attempts = retry_budget + 1
    started_at = time.monotonic()
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    total_turns = 0
    last_result: KerminalRunResult | None = None

    for attempt in range(1, max_attempts + 1):
        attempt_max_time = _attempt_max_time(max_time, started_at)
        if attempt_max_time == 0:
            return KerminalRunResult(
                submitted=False,
                solution_path=None,
                turns_used=total_turns,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_tokens=total_cache_creation_tokens,
                cache_read_tokens=total_cache_read_tokens,
                error="timeout_exceeded",
            )

        result = await _run_kerminal_agent_once_async(
            model_id=model_id,
            sandbox=sandbox,
            system_prompt=system_prompt,
            initial_user_message=initial_user_message,
            max_turns=max_turns,
            max_time=attempt_max_time,
            is_metal=is_metal,
        )
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        total_cache_creation_tokens += result.cache_creation_tokens
        total_cache_read_tokens += result.cache_read_tokens
        total_turns += result.turns_used
        last_result = _with_accumulated_usage(
            result,
            turns_used=total_turns,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_creation_tokens=total_cache_creation_tokens,
            cache_read_tokens=total_cache_read_tokens,
        )

        if not result.error:
            return last_result
        if attempt >= max_attempts or not _is_retryable_kerminal_error(result.error):
            return last_result

        _record_kerminal_retry(attempt, max_attempts, result.error)
        delay = _network_retry_delay_seconds(attempt)
        if delay > 0:
            await asyncio.sleep(delay)

    if last_result is not None:
        return last_result

    return KerminalRunResult(
        submitted=False,
        solution_path=None,
        turns_used=0,
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        error="kerminal_retry_failed_without_result",
    )


async def _run_kerminal_agent_once_async(
    *,
    model_id: str,
    sandbox,
    system_prompt: str,
    initial_user_message: str,
    max_turns: int,
    max_time: Optional[int],
    is_metal: bool,
) -> KerminalRunResult:
    try:
        from kerminal import (
            ClientLogEntry,
            ClientOptions,
            ErrorEvent,
            KerminalClient,
            NewConversationOptions,
            SendUserTurnOptions,
            StreamErrorEvent,
            TaskCompleteEvent,
            TokenCountEvent,
            TurnAbortedEvent,
        )
    except Exception as exc:
        return KerminalRunResult(
            submitted=False,
            solution_path=None,
            turns_used=0,
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            error=f"failed to import kerminal SDK: {exc}",
        )

    artifact_dir = _artifact_dir()
    raw_log = artifact_dir / "kerminal_rpc.jsonl" if artifact_dir else None
    event_log = artifact_dir / "kerminal_events.jsonl" if artifact_dir else None

    def append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")

    def on_log(entry: ClientLogEntry) -> None:
        append_jsonl(
            raw_log,
            {
                "direction": entry.direction,
                "kind": entry.kind,
                "message": entry.message,
            },
        )

    workspace = _sandbox_workspace(sandbox)
    model = None if model_id == "default" else model_id
    effective_max_turns = max(1, int(max_turns or 1))
    deadline = time.monotonic() + float(max_time if max_time is not None else effective_max_turns * 60)
    outer_turns_sent = 0
    submitted = False
    solution_path: Optional[str] = None
    state: dict[str, Any] = {
        "conversation_id": None,
        "error": None,
        "task_complete": False,
        "turn_done": None,
        "active_turn_id": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
    }
    seen_turn_ids: set[str] = set()
    completed_turn_ids: set[str] = set()

    client = await _start_kerminal_client_in_workspace(
        KerminalClient,
        ClientOptions(
            binary_path=os.environ.get("KERMINAL_BIN", "kerminal"),
            env=_client_env(),
            on_log=on_log,
        ),
        workspace,
    )

    async def accept_approval(request: dict[str, Any]) -> None:
        try:
            await client.respond_approval(request["id"], "approved")
        except Exception as exc:
            state["error"] = f"approval response failed: {exc}"
            _signal_turn_done(state)

    def on_approval(request: dict[str, Any]) -> None:
        append_jsonl(event_log, {"type": "approval_request", "request": request})
        asyncio.create_task(accept_approval(request))

    def on_event(msg: Any, conversation_id: str, turn_id: str) -> None:
        append_jsonl(
            event_log,
            {
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "type": getattr(msg, "type", "unknown"),
                "raw": getattr(msg, "raw", None),
            },
        )
        active_conversation_id = state.get("conversation_id")
        if active_conversation_id is not None and conversation_id != active_conversation_id:
            return

        event_type = getattr(msg, "type", None)
        if event_type == "task_started":
            seen_turn_ids.add(turn_id)
            if state.get("active_turn_id") is None:
                state["active_turn_id"] = turn_id
            return
        if isinstance(msg, TokenCountEvent):
            seen_turn_ids.add(turn_id)
            info = msg.info or {}
            total = info.get("total_token_usage", {}) if isinstance(info, dict) else {}
            state["input_tokens"] = int(total.get("input_tokens") or 0)
            state["output_tokens"] = int(total.get("output_tokens") or 0)
            state["cache_creation_tokens"] = int(total.get("cache_creation_input_tokens") or 0)
            state["cache_read_tokens"] = int(total.get("cache_read_input_tokens") or 0)
            return
        if isinstance(msg, TaskCompleteEvent):
            if turn_id in completed_turn_ids:
                return
            completed_turn_ids.add(turn_id)
            state["task_complete"] = True
            _signal_turn_done(state)
            return
        if isinstance(msg, ErrorEvent):
            state["error"] = msg.message
            _signal_turn_done(state)
            return
        if isinstance(msg, StreamErrorEvent):
            state["error"] = f"stream_error: {msg.message}"
            _signal_turn_done(state)
            return
        if isinstance(msg, TurnAbortedEvent):
            state["error"] = f"turn_aborted: {msg.reason}"
            _signal_turn_done(state)

    unsubscribe_event = client.on_event_msg(on_event)
    unsubscribe_approval = client.on_approval(on_approval)
    try:
        await client.initialize(
            {"name": "kernelbench-v3", "title": "KernelBench v3", "version": "0.1.0"}
        )
        if api_key := _api_key():
            await client.login_api_key(api_key)
        convo = await client.new_conversation(
            NewConversationOptions(
                model=model,
                cwd=str(workspace),
                approval_policy="never",
            )
        )
        conversation_id = convo["conversationId"]
        state["conversation_id"] = conversation_id
        await client.add_conversation_listener(conversation_id, experimental_raw_events=False)

        mapped_system_prompt = _map_prompt_workspace(system_prompt, workspace)
        mapped_initial_user_message = _map_prompt_workspace(initial_user_message, workspace)
        next_turn_text = _compose_prompt(mapped_system_prompt, mapped_initial_user_message)
        kerminal_model = convo.get("model") or model or "default"

        for outer_turn in range(1, effective_max_turns + 1):
            remaining = _remaining_seconds(deadline)
            if remaining <= 0:
                state["error"] = "timeout_exceeded"
                break

            if outer_turn == 1:
                append_jsonl(
                    event_log,
                    {
                        "type": "prompt_payload",
                        "outer_turn": outer_turn,
                        "system_prompt": mapped_system_prompt,
                        "initial_user_message": mapped_initial_user_message,
                        "user_turn_text": next_turn_text,
                    },
                )
            else:
                append_jsonl(
                    event_log,
                    {
                        "type": "feedback_payload",
                        "outer_turn": outer_turn,
                        "user_turn_text": next_turn_text,
                    },
                )

            state["task_complete"] = False
            state["active_turn_id"] = None
            state["turn_done"] = asyncio.Event()
            items = [{"type": "text", "data": {"text": next_turn_text}}]
            await client.send_user_turn(
                SendUserTurnOptions(
                    conversation_id=conversation_id,
                    items=items,
                    cwd=str(workspace),
                    approval_policy="never",
                    model=kerminal_model,
                )
            )
            outer_turns_sent = outer_turn

            await asyncio.wait_for(state["turn_done"].wait(), timeout=remaining)
            state["turn_done"] = None

            if state["error"]:
                break
            if not state["task_complete"]:
                state["error"] = "turn_finished_without_task_complete"
                break

            verification = _verify_solution(
                sandbox=sandbox,
                is_metal=is_metal,
                turn=outer_turn,
                deadline=deadline,
                artifact_dir=artifact_dir,
            )
            if verification.submitted:
                submitted = True
                solution_path = verification.solution_path
                break
            if verification.error:
                state["error"] = verification.error
                break
            if outer_turn >= effective_max_turns:
                state["error"] = "max_turns_exceeded"
                break
            next_turn_text = verification.feedback or _missing_solution_feedback(workspace)
    except asyncio.TimeoutError:
        state["error"] = "timeout_exceeded"
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        state["turn_done"] = None
        unsubscribe_event()
        unsubscribe_approval()
        try:
            await client.close()
        except Exception:
            pass

    return KerminalRunResult(
        submitted=submitted,
        solution_path=solution_path,
        turns_used=outer_turns_sent or len(seen_turn_ids),
        input_tokens=int(state["input_tokens"]),
        output_tokens=int(state["output_tokens"]),
        cache_creation_tokens=int(state["cache_creation_tokens"]),
        cache_read_tokens=int(state["cache_read_tokens"]),
        error=state["error"],
    )


def _artifact_dir() -> Path | None:
    raw = os.environ.get("KB_TURN_ARTIFACT_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _client_env() -> dict[str, str]:
    env = {}
    pythonpath = os.environ.get("PYTHONPATH", "")
    sdk_path = str(KERMINAL_SDK_SRC)
    if KERMINAL_SDK_SRC.exists() and sdk_path not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{sdk_path}:{pythonpath}" if pythonpath else sdk_path
    return env


def _api_key() -> str | None:
    api_key = os.environ.get("KERMINAL_API_KEY")
    return api_key.strip() if api_key and api_key.strip() else None


def _network_retry_budget() -> int:
    value = os.environ.get("KERMINAL_NETWORK_RETRIES")
    if value is None or not value.strip():
        return DEFAULT_KERMINAL_NETWORK_RETRIES
    try:
        return max(0, int(value))
    except ValueError:
        return DEFAULT_KERMINAL_NETWORK_RETRIES


def _network_retry_delay_seconds(attempt: int) -> float:
    value = os.environ.get("KERMINAL_NETWORK_RETRY_DELAY_SECONDS")
    if value is not None and value.strip():
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    return min(10.0, float(2 ** max(0, attempt - 1)))


def _attempt_max_time(max_time: Optional[int], started_at: float) -> Optional[int]:
    if max_time is None:
        return None
    remaining = int(max_time - (time.monotonic() - started_at))
    return max(0, remaining)


def _with_accumulated_usage(
    result: KerminalRunResult,
    *,
    turns_used: int,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> KerminalRunResult:
    return KerminalRunResult(
        submitted=result.submitted,
        solution_path=result.solution_path,
        turns_used=turns_used,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        error=result.error,
    )


def _is_retryable_kerminal_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.lower()
    retryable_markers = (
        "stream_error",
        "network",
        "connection",
        "connect",
        "econnreset",
        "econnrefused",
        "etimedout",
        "socket",
        "tls",
        "timed out",
        "temporarily unavailable",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
        "gateway",
        "service unavailable",
        "kerminal process exited",
        "broken pipe",
        "connection reset",
    )
    return any(marker in normalized for marker in retryable_markers)


def _record_kerminal_retry(attempt: int, max_attempts: int, error: str) -> None:
    artifact_dir = _artifact_dir()
    event_log = artifact_dir / "kerminal_events.jsonl" if artifact_dir else None
    payload = {
        "type": "kerminal_retry",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error": error,
    }
    if event_log is not None:
        try:
            with open(event_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")
        except Exception:
            pass
    print(
        f"[Kerminal retry] attempt {attempt}/{max_attempts} failed with retryable error: {error}",
        flush=True,
    )


def _sandbox_workspace(sandbox) -> Path:
    workspace = getattr(sandbox, "workspace_path", None)
    if workspace is None:
        raise RuntimeError("Kerminal runner requires a local sandbox with workspace_path")
    return Path(workspace)


def _compose_prompt(system_prompt: str, initial_user_message: str) -> str:
    return f"{system_prompt}\n\n{initial_user_message}"


def _map_prompt_workspace(text: str, workspace: Path) -> str:
    return text.replace("/workspace", str(workspace))


async def _start_kerminal_client_in_workspace(
    client_cls: Any,
    options: Any,
    workspace: Path,
) -> Any:
    previous_cwd = Path.cwd()
    os.chdir(workspace)
    try:
        return await client_cls.start(options)
    finally:
        os.chdir(previous_cwd)


def _signal_turn_done(state: dict[str, Any]) -> None:
    turn_done = state.get("turn_done")
    if isinstance(turn_done, asyncio.Event):
        turn_done.set()


def _remaining_seconds(deadline: float) -> float:
    return deadline - time.monotonic()


def _bounded_timeout(deadline: float, default_timeout: int) -> Optional[int]:
    remaining = _remaining_seconds(deadline)
    if remaining <= 0:
        return None
    return max(1, min(default_timeout, int(remaining)))


def _format_command_result(cmd: str, cmd_result: dict[str, Any]) -> str:
    return (
        f"command: {cmd}\n"
        f"return_code: {cmd_result.get('returncode')}\n"
        f"stdout:\n{cmd_result.get('stdout', '')}\n"
        f"stderr:\n{cmd_result.get('stderr', '')}\n"
    )


def _write_turn_artifact(
    artifact_dir: Path | None,
    turn: int,
    suffix: str,
    content: str,
) -> None:
    if artifact_dir is None:
        return
    try:
        path = artifact_dir / f"turn_{turn}_{suffix}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


def _verify_solution(
    *,
    sandbox,
    is_metal: bool,
    turn: int,
    deadline: float,
    artifact_dir: Path | None,
) -> _VerificationResult:
    workspace = _sandbox_workspace(sandbox)
    if not sandbox.file_exists("solution.py"):
        return _VerificationResult(
            submitted=False,
            solution_path=None,
            status="missing_solution",
            feedback=_missing_solution_feedback(workspace),
        )

    compile_checks = [
        ('python -c "from solution import Model; m = Model(); print(\'OK\')"', "Model import check OK"),
        ('python -c "import solution; print(\'OK\')"', "module import check OK"),
    ]
    compile_logs: list[str] = []
    compile_ok = False
    for compile_cmd, _success_label in compile_checks:
        timeout = _bounded_timeout(deadline, 120)
        if timeout is None:
            return _VerificationResult(
                submitted=False,
                solution_path=None,
                status="timeout",
                error="timeout_exceeded",
            )
        compile_result = sandbox.run_command(compile_cmd, timeout=timeout)
        compile_logs.append(_format_command_result(compile_cmd, compile_result))
        if compile_result.get("returncode") == 0 and "OK" in (compile_result.get("stdout") or ""):
            compile_ok = True
            break

    compile_log = "\n\n".join(compile_logs)
    _write_turn_artifact(artifact_dir, turn, "compile.log", compile_log)
    if not compile_ok:
        return _VerificationResult(
            submitted=False,
            solution_path=None,
            status="compile_failed",
            feedback=_command_failure_feedback(
                "The current solution.py failed to import or compile.",
                compile_log,
            ),
        )

    check_cmd = self_check_command(is_metal)
    timeout = _bounded_timeout(deadline, 300)
    if timeout is None:
        return _VerificationResult(
            submitted=False,
            solution_path=None,
            status="timeout",
            error="timeout_exceeded",
        )
    check_result = sandbox.run_command(check_cmd, timeout=timeout)
    check_log = _format_command_result(check_cmd, check_result)
    _write_turn_artifact(artifact_dir, turn, "self_check.log", check_log)
    if check_result.get("returncode") == 0 and _self_check_passed(check_result.get("stdout") or ""):
        return _VerificationResult(
            submitted=True,
            solution_path="solution.py",
            status="self_check_passed",
        )

    return _VerificationResult(
        submitted=False,
        solution_path=None,
        status="self_check_failed",
        feedback=_command_failure_feedback(
            "The current solution.py failed the required correctness self-check.",
            check_log,
        ),
    )


def _self_check_passed(stdout: str) -> bool:
    return any(line.strip() == "PASS" for line in stdout.splitlines())


def _missing_solution_feedback(workspace: Path) -> str:
    return (
        "No solution.py was found in the current working directory for this run "
        f"({workspace}). Continue in the same workspace and create or update a file "
        "named solution.py there. "
        "Run the required correctness self-check again before finishing."
    )


def _command_failure_feedback(summary: str, command_log: str) -> str:
    return (
        f"{summary}\n\n"
        "Fix solution.py in the current working directory for this same run, then run "
        "the required correctness self-check again before finishing.\n\n"
        "Command output:\n"
        "```\n"
        f"{_truncate_for_feedback(command_log)}\n"
        "```"
    )


def _truncate_for_feedback(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    keep = max(1, limit // 2)
    omitted = len(text) - (keep * 2)
    return f"{text[:keep]}\n... (truncated {omitted} chars) ...\n{text[-keep:]}"
