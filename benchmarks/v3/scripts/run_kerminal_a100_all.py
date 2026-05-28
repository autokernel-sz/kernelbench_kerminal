#!/usr/bin/env python3
"""Run the full local A100 KernelBench evaluation with Kerminal."""

from __future__ import annotations

from run_kerminal_local_all import main


if __name__ == "__main__":
    raise SystemExit(main(default_hardware="a100_local"))
