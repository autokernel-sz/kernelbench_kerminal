#!/usr/bin/env python3
"""Run a full local NVIDIA KernelBench evaluation with Kerminal."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
KERMINAL_SDK_SRC = Path("/workspace/kerminal/sdk/python/src")
DEFAULT_KERMINAL_BIN_DIR = Path.home() / ".local" / "bin"


def _kerminal_home(env: dict[str, str]) -> Path:
    return Path(env.get("KERMINAL_HOME", Path.home() / ".kerminal")).expanduser()


def _has_kerminal_config_auth(env: dict[str, str]) -> bool:
    home = _kerminal_home(env)
    env_file = home / ".env"
    if env_file.exists() and "KERMINAL_API_KEY=" in env_file.read_text(encoding="utf-8", errors="ignore"):
        return True

    config = home / "config.toml"
    if not config.exists():
        return False
    config_text = config.read_text(encoding="utf-8", errors="ignore")
    return "experimental_bearer_token" in config_text or "api_key" in config_text


def _env_for_kerminal(require_api_key: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    api_key = env.get("KERMINAL_API_KEY", "").strip()
    has_config_auth = _has_kerminal_config_auth(env)
    if require_api_key and not api_key and not has_config_auth:
        api_key = getpass.getpass("KERMINAL_API_KEY: ").strip()
    if require_api_key and not api_key and not has_config_auth:
        raise SystemExit("KERMINAL_API_KEY is required unless ~/.kerminal config already has auth.")
    if api_key:
        env["KERMINAL_API_KEY"] = api_key

    if KERMINAL_SDK_SRC.exists():
        pythonpath = env.get("PYTHONPATH", "")
        parts = [p for p in pythonpath.split(os.pathsep) if p]
        sdk_path = str(KERMINAL_SDK_SRC)
        if sdk_path not in parts:
            env["PYTHONPATH"] = os.pathsep.join([sdk_path, *parts])

    if DEFAULT_KERMINAL_BIN_DIR.exists():
        path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
        bin_dir = str(DEFAULT_KERMINAL_BIN_DIR)
        if bin_dir not in path_parts:
            env["PATH"] = os.pathsep.join([bin_dir, *path_parts])

    return env


def _build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "bench.py",
        "run",
        args.hardware,
        "--models",
        args.model,
        "--levels",
        args.levels,
        "--workers",
        str(args.workers),
    ]
    if args.problems_per_level is not None:
        cmd.extend(["--problems-per-level", str(args.problems_per_level)])
    if args.problem:
        cmd.extend(["--problem", args.problem])
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.judge_model:
        cmd.extend(["--judge-model", args.judge_model])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main(argv: list[str] | None = None, default_hardware: str = "h100_local") -> int:
    parser = argparse.ArgumentParser(
        description="Run Kerminal on a full local NVIDIA KernelBench target."
    )
    parser.add_argument("--hardware", default=default_hardware, help="Benchmark hardware target")
    parser.add_argument("--model", default="kerminal/default", help="KernelBench model key")
    parser.add_argument("--levels", default="1,2,3,4", help="Comma-separated levels")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers. Keep this at 1 until Kerminal cwd startup is made parallel-safe.",
    )
    parser.add_argument("--problems-per-level", type=int, default=None)
    parser.add_argument("--problem", default=None, help="Run one problem by filename substring")
    parser.add_argument("--resume", default=None, help="Resume from an outputs/batch_eval/run_* directory")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.workers != 1:
        print(
            "Warning: Kerminal local cwd startup is currently intended for serial runs; "
            f"requested --workers {args.workers}.",
            file=sys.stderr,
        )

    env = _env_for_kerminal(require_api_key=not args.dry_run)
    cmd = _build_command(args)

    print("Running from:", REPO_ROOT)
    print("Command:", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
