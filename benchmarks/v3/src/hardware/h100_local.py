"""Local H100 hardware target — runs directly on the host GPU."""

from src.hardware import HardwareTarget, register


@register("h100_local")
class H100LocalTarget(HardwareTarget):
    name = "h100_local"
    display_name = "H100 Local"
    gpu_sku = "H100"
    vram_gb = 80
    problem_dirs = ["level1", "level2", "level3", "level4", "tile_specialized"]
    exclude_problems = ["9_FP4_BlockScaled_Matmul.py"]

    def create_sandbox(self, problem_code: str):
        from src.agent.local_sandbox import LocalSandbox, LocalSandboxConfig

        return LocalSandbox(problem_code, LocalSandboxConfig(timeout=300))
