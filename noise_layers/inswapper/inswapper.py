import atexit
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path(r"D:\zotero_translate\python.exe")
DEFAULT_ONNX = PROJECT_ROOT / "noise_layers" / "InSwapper-128" / "inswapper_128.onnx"
DEFAULT_RECOGNIZER = (
    PROJECT_ROOT / "noise_layers" / "insightface_models" / "w600k_r50.onnx"
)
DEFAULT_DETECTOR = (
    PROJECT_ROOT / "noise_layers" / "insightface_models" / "det_10g.onnx"
)
DEFAULT_WORKER = PROJECT_ROOT / "noise_layers" / "inswapper" / "onnx_worker.py"


class InSwapper(nn.Module):
    def __init__(
        self,
        resolution=128,
        onnx_path=None,
        recognizer_path=None,
        detector_path=None,
        python_executable=None,
        worker_script=None,
    ):
        super(InSwapper, self).__init__()
        self.resolution = int(resolution)
        self.onnx_path = Path(onnx_path or DEFAULT_ONNX).resolve()
        self.recognizer_path = Path(recognizer_path or DEFAULT_RECOGNIZER).resolve()
        self.detector_path = Path(detector_path or DEFAULT_DETECTOR).resolve()
        self.python_executable = Path(python_executable or DEFAULT_PYTHON).resolve()
        self.worker_script = Path(worker_script or DEFAULT_WORKER).resolve()

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"InSwapper ONNX not found: {self.onnx_path}")
        if not self.recognizer_path.exists():
            raise FileNotFoundError(
                f"InsightFace recognizer not found: {self.recognizer_path}"
            )
        if not self.detector_path.exists():
            raise FileNotFoundError(
                f"InsightFace detector not found: {self.detector_path}"
            )
        if not self.python_executable.exists():
            raise FileNotFoundError(f"ONNX Python not found: {self.python_executable}")
        if not self.worker_script.exists():
            raise FileNotFoundError(f"ONNX worker not found: {self.worker_script}")

        self._worker = None
        self._stderr_handle = None
        atexit.register(self.close)

    def _child_environment(self):
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["OMP_NUM_THREADS"] = "4"
        environment["PATH"] = (
            str(self.python_executable.parent)
            + os.pathsep
            + environment.get("PATH", "")
        )
        return environment

    def _ensure_worker(self):
        if self._worker is not None and self._worker.poll() is None:
            return

        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "inswapper_worker_stderr.log"
        self._stderr_handle = open(log_path, "a", encoding="utf-8")
        self._worker = subprocess.Popen(
            [
                str(self.python_executable),
                str(self.worker_script),
                "--model",
                str(self.onnx_path),
                "--recognizer",
                str(self.recognizer_path),
                "--detector",
                str(self.detector_path),
            ],
            cwd=str(PROJECT_ROOT),
            env=self._child_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        ready = self._worker.stdout.readline()
        if not ready:
            self.close()
            raise RuntimeError(
                "InSwapper ONNX worker failed to start; see "
                f"{log_path}"
            )
        message = json.loads(ready)
        if not message.get("ready"):
            self.close()
            raise RuntimeError(f"unexpected InSwapper worker start message: {message}")

    def _worker_request(self, payload, operation):
        self._ensure_worker()
        temp_root = PROJECT_ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="inswapper_", dir=str(temp_root)) as directory:
            input_path = os.path.join(directory, "input.npz")
            output_path = os.path.join(directory, "output.npy")
            np.savez(input_path, **payload)
            request = json.dumps(
                {
                    "input_npz": input_path,
                    "output_npy": output_path,
                    "operation": operation,
                }
            )
            self._worker.stdin.write(request + "\n")
            self._worker.stdin.flush()
            response_line = self._worker.stdout.readline()
            if not response_line:
                log_path = PROJECT_ROOT / "logs" / "inswapper_worker_stderr.log"
                tail = ""
                if log_path.exists():
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                self.close()
                raise RuntimeError(f"InSwapper ONNX worker stopped.\n{tail}")
            response = json.loads(response_line)
            if not response.get("ok"):
                raise RuntimeError(
                    "InSwapper ONNX request failed: "
                    + response.get("traceback", response.get("error", "unknown error"))
                )
            return np.load(output_path)

    def _to_uint8_nhwc(self, image):
        return (
            ((image + 1.0) / 2.0)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(th.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

    def embed(self, image):
        with th.no_grad():
            images = self._to_uint8_nhwc(image)
            output = self._worker_request({"images": images}, "embed")
            return th.from_numpy(output).to(device=image.device, dtype=image.dtype)

    def forward(self, image_cover_mask):
        with th.no_grad():
            image = image_cover_mask[0]
            cover_image = image_cover_mask[1]
            if image.shape[0] != cover_image.shape[0]:
                raise ValueError("image and identity-source batch sizes differ")

            if image.shape[0] == 1:
                if th.allclose(image, cover_image):
                    raise ValueError(
                        "InSwapper needs a different identity source; "
                        "batch=1 self-swap is disabled"
                    )
                identity = cover_image
            else:
                identity = th.roll(cover_image, shifts=1, dims=0)

            target = self._to_uint8_nhwc(image)
            identity = self._to_uint8_nhwc(identity)
            output = self._worker_request(
                {"target": target, "identity": identity},
                "swap",
            )
            output = th.from_numpy(output).to(
                device=image.device,
                dtype=image.dtype,
            )
            output = output.permute(0, 3, 1, 2).mul(2.0).sub(1.0)
            if output.shape[-2:] != image.shape[-2:]:
                output = F.interpolate(
                    output,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return output.contiguous()

    def close(self):
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.poll() is None:
            try:
                if worker.stdin is not None:
                    worker.stdin.write(json.dumps({"close": True}) + "\n")
                    worker.stdin.flush()
            except Exception:
                pass
            try:
                worker.wait(timeout=3)
            except Exception:
                worker.terminate()
        self._worker = None
        handle = getattr(self, "_stderr_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        self._stderr_handle = None
