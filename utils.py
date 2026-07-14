import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

_FONTM = cv2.FONT_HERSHEY_SIMPLEX


class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last_time = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        return self.fps


def _get_gpu_util_smi() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1,
        )
        if result.returncode == 0:
            output = result.stdout.strip().split("\n")[0]
            if output:
                return float(output)
    except Exception:
        pass
    return -1.0


class ResourceMonitor:
    def __init__(self, interval: int = 5):
        self._proc = psutil.Process()
        self._interval = interval
        self._n = 0
        self._cache: dict = {}
        self._using_gpu = torch.cuda.is_available()

        self._nvml_ok = False
        if self._using_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._pynvml = pynvml
                self._nvml_ok = True
            except Exception:
                pass  

    def sample(self, fps: float = 0.0) -> dict:
        self._n += 1
        if self._n % self._interval != 0 and self._cache:
            self._cache["fps"] = fps
            return self._cache

        mem = self._proc.memory_info()
        cpu_pct = self._proc.cpu_percent(interval=None)

        data: dict = {
            "using_gpu": self._using_gpu,
            "ram_mb": mem.rss / 1024 / 1024,
            "cores_used": round(cpu_pct / 100, 1),
            "fps": fps,
        }

        if self._using_gpu:
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)
            data["vram_mb"] = allocated / 1024 / 1024
            data["vram_pct"] = allocated / props.total_memory * 100

            gpu_util = -1.0
            if self._nvml_ok:
                try:
                    gpu_util = self._pynvml.nvmlDeviceGetUtilizationRates(
                        self._nvml_handle).gpu
                except Exception:
                    gpu_util = -1.0
            if gpu_util < 0:
                gpu_util = _get_gpu_util_smi()
            data["gpu_util"] = gpu_util if gpu_util >= 0 else None

        self._cache = data
        return data


def draw_resources(frame: np.ndarray, res: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    using_gpu = res.get("using_gpu", False)
    fps = res.get("fps", 0.0)

    lines = [(f"FPS  {fps:.1f}", (0, 220, 80))]

    if using_gpu:
        gpu_util = res.get("gpu_util")
        if gpu_util is not None:
            lines.append((f"GPU  {gpu_util:.0f}%", (0, 190, 255)))
        lines.append((f"RAM  {res.get('ram_mb', 0):.0f}MB", (0, 190, 255)))
        lines.append((f"VRAM {res.get('vram_mb', 0):.0f}MB", (255, 180, 0)))
    else:
        lines.append((f"CPU  {res.get('cores_used', 0)} core", (0, 190, 255)))
        lines.append((f"RAM  {res.get('ram_mb', 0):.0f}MB", (0, 190, 255)))

    pad, lh, pw = 6, 18, 130
    ph = len(lines) * lh + pad
    px = w - pw - 8
    py = 8

    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 60, 60), 1)

    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text,
            (px + pad, py + pad + (i + 1) * lh - 4),
            _FONTM, 0.42, color, 1, cv2.LINE_AA,
        )

    return frame


def make_output_path(output_dir: str, source_name: str, suffix: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    return out_dir / f"{stem}_detected{suffix}"


def print_detections(detections, label: str = ""):
    if not detections:
        print(f"{label} -> no detection")
        return
    summary = ", ".join(f"{d['class']} ({d['confidence'] * 100:.1f}%)" for d in detections)
    print(f"{label} -> {summary}")