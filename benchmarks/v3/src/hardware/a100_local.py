"""Local A100 hardware target — runs directly on the host GPU."""

from src.hardware import HardwareTarget, register


@register("a100_local")
class A100LocalTarget(HardwareTarget):
    name = "a100_local"
    display_name = "A100 Local"
    gpu_sku = "A100"
    vram_gb = 80
    problem_dirs = ["level1", "level2", "level3", "level4", "tile_specialized"]
    exclude_problems = [
        "4_FP8_Matmul.py",
        "9_FP4_BlockScaled_Matmul.py",
    ]

    def create_sandbox(self, problem_code: str):
        from src.agent.local_sandbox import LocalSandbox, LocalSandboxConfig

        return LocalSandbox(problem_code, LocalSandboxConfig(timeout=300))
