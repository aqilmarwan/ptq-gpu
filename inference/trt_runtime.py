"""TensorRT execution primitives for the serving plane.

This is the HF-free replacement for the diffusers runtime: the heavy forward
passes (text encoders, UNet, VAE decoder) run as prebuilt TensorRT engines, and
the scheduler is a vendored Euler-discrete implementation so there is no
`diffusers` dependency at serving time. The only third-party pieces kept are
`torch` (numeric framework + device buffers) and `transformers.CLIPTokenizer`,
which we load from the bundled vocab/merges files -- no Hugging Face Hub access,
no HF token.

Nothing here imports at module load beyond torch-free stdlib; `torch`,
`tensorrt`, and `transformers` are imported lazily so the demo plane and CI stay
GPU/HF-free.

NOTE: the TRT execution and Euler math are exercised only on a real GPU with real
engines. They are written to the TensorRT 10 API and the diffusers Euler defaults
but must be validated on-device before production.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only; never imported at runtime on the demo plane
    import torch

log = logging.getLogger("studio.trt")

# Engine bundle layout (one directory per variant, fetched from S3 to ENGINE_DIR):
#   <id>/text_encoder.plan   text_encoder_2.plan   unet.plan   vae_decoder.plan
#   <id>/tokenizer/          tokenizer_2/          metadata.json
ENGINE_FILES = ("text_encoder", "text_encoder_2", "unet", "vae_decoder")


# --------------------------------------------------------------------------- #
# TensorRT engine wrapper
# --------------------------------------------------------------------------- #


class TRTEngine:
    """One deserialized TensorRT engine + execution context.

    ``infer`` binds torch CUDA tensors directly (via ``data_ptr``), runs on the
    current CUDA stream, and returns freshly allocated output tensors.
    """

    _LOGGER = None  # a single shared trt.Logger

    def __init__(self, plan_path: Path):
        import tensorrt as trt

        if TRTEngine._LOGGER is None:
            TRTEngine._LOGGER = trt.Logger(trt.Logger.WARNING)

        runtime = trt.Runtime(TRTEngine._LOGGER)
        self.engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine: {plan_path}")
        self.context = self.engine.create_execution_context()
        self._trt = trt

        self.inputs = [n for n in self._tensor_names() if self._is_input(n)]
        self.outputs = [n for n in self._tensor_names() if not self._is_input(n)]

    def _tensor_names(self):
        return [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

    def _is_input(self, name: str) -> bool:
        return self.engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT

    def infer(self, feeds: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
        import torch

        stream = torch.cuda.current_stream().cuda_stream

        for name in self.inputs:
            t = feeds[name].contiguous()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        results: dict[str, torch.Tensor] = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _torch_dtype(self.engine.get_tensor_dtype(name), torch)
            out = torch.empty(shape, dtype=dtype, device="cuda")
            results[name] = out
            self.context.set_tensor_address(name, out.data_ptr())

        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TRT execute_async_v3 failed")
        torch.cuda.current_stream().synchronize()
        return results


def _torch_dtype(trt_dtype, torch):
    import tensorrt as trt

    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
    }.get(trt_dtype, torch.float16)


# --------------------------------------------------------------------------- #
# Vendored Euler-discrete scheduler (SDXL defaults, no diffusers)
# --------------------------------------------------------------------------- #


class EulerScheduler:
    """Euler-discrete scheduler matching diffusers' SDXL defaults.

    scaled_linear betas, 1000 train steps, epsilon prediction. Enough for the
    serving loop; training-time features are intentionally omitted.
    """

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        import torch

        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self._train_sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, steps: int, device: str = "cuda"):
        import torch

        idx = torch.linspace(0, self.num_train_timesteps - 1, steps).flip(0)
        low = idx.floor().long()
        high = idx.ceil().long()
        frac = idx - idx.floor()
        sigmas = (1 - frac) * self._train_sigmas[low] + frac * self._train_sigmas[high]
        self.sigmas = torch.cat([sigmas, torch.zeros(1)]).to(device)
        self.timesteps = (idx * 1.0).to(device)  # continuous t passed to the UNet
        return self.timesteps

    @property
    def init_noise_sigma(self) -> float:
        return float(self.sigmas.max()) if self.sigmas is not None else float(self._train_sigmas.max())

    def scale_model_input(self, sample, step_index: int):
        sigma = self.sigmas[step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

    def step(self, model_output, step_index: int, sample):
        """One Euler update. model_output is the epsilon prediction."""
        sigma = self.sigmas[step_index]
        pred_original = sample - sigma * model_output
        derivative = (sample - pred_original) / sigma
        dt = self.sigmas[step_index + 1] - sigma
        return sample + derivative * dt


# --------------------------------------------------------------------------- #
# Tokenizers (transformers, loaded offline from the bundle)
# --------------------------------------------------------------------------- #


def load_tokenizers(bundle_dir: Path):
    """Load the two CLIP tokenizers from bundled files -- offline, no HF Hub."""
    from transformers import CLIPTokenizer

    tok1 = CLIPTokenizer.from_pretrained(str(bundle_dir / "tokenizer"), local_files_only=True)
    tok2 = CLIPTokenizer.from_pretrained(str(bundle_dir / "tokenizer_2"), local_files_only=True)
    return tok1, tok2
