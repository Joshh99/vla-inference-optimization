"""Configurable SmolVLA inference with an optional low-latency path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference


@dataclass(frozen=True)
class InferenceOptions:
    denoising_steps: int = 8
    reuse_window: int = 10
    compile_forward: bool = True
    fp16_autocast: bool = False
    output_action_dim: int = 7

    @classmethod
    def reference(cls) -> "InferenceOptions":
        return cls(denoising_steps=10, reuse_window=1, compile_forward=False)


class SmolVLARunner:
    def __init__(self, options: InferenceOptions | None = None) -> None:
        self.options = options or InferenceOptions()
        self.policy: SmolVLAPolicy | None = None
        self.device: torch.device | None = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.model_action_dim = 0
        self._cached_prediction: np.ndarray | None = None
        self._reuse_count = 0

    @property
    def optimized(self) -> bool:
        return self.options != InferenceOptions.reference()

    def initialize(self, model_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_float32_matmul_precision("high")

        policy = SmolVLAPolicy.from_pretrained(model_path).to(self.device).eval()
        policy.requires_grad_(False)
        policy.config.num_steps = self.options.denoising_steps
        policy.model.config.num_steps = self.options.denoising_steps
        self.model_action_dim = policy.config.action_feature.shape[0]

        if self.optimized:
            language_head = getattr(policy.model.vlm_with_expert.vlm, "lm_head", None)
            if language_head is not None:
                del policy.model.vlm_with_expert.vlm.lm_head

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        if self.options.compile_forward:
            policy.predict_action_chunk = torch.compile(
                policy.predict_action_chunk,
                mode="reduce-overhead",
            )

        self.policy = policy
        self._warmup()

    def reset(self) -> None:
        self._cached_prediction = None
        self._reuse_count = 0
        if self.policy is not None:
            self.policy.reset()

    def predict(self, episode: dict[str, Any]) -> np.ndarray:
        if self.policy is None or self.device is None:
            raise RuntimeError("initialize() must be called before predict()")

        if self._cached_prediction is not None and self._reuse_count < self.options.reuse_window:
            self._reuse_count += 1
            return self._cached_prediction

        observation = prepare_observation_for_inference(
            self._observation(episode),
            self.device,
            task=episode["instruction"],
        )
        observation = self.preprocessor(observation)

        with torch.inference_mode():
            if self.options.fp16_autocast and self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    action_tensor = self.policy.predict_action_chunk(observation)
            else:
                action_tensor = self.policy.predict_action_chunk(observation)

        result = self._to_numpy(action_tensor)
        if self.options.reuse_window > 1:
            self._cached_prediction = result
            self._reuse_count = 1
        return result

    def _to_numpy(self, action_tensor: torch.Tensor) -> np.ndarray:
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)

        if self.optimized:
            actions = action_tensor.squeeze(0)[:, : self.model_action_dim]
        else:
            actions = torch.stack(
                [self.postprocessor(action_tensor[:, index, :]) for index in range(action_tensor.shape[1])],
                dim=1,
            ).squeeze(0)

        array = actions.float().cpu().numpy().astype(np.float32, copy=False)
        width = self.options.output_action_dim
        if array.shape[1] < width:
            array = np.pad(array, ((0, 0), (0, width - array.shape[1])))
        return array[:, :width].astype(np.float32, copy=False)

    def _warmup(self) -> None:
        sample = {
            "images": {
                "camera1": np.zeros((512, 512, 3), dtype=np.uint8),
                "camera2": np.zeros((512, 512, 3), dtype=np.uint8),
            },
            "state": np.zeros(8, dtype=np.float32),
            "instruction": "warmup",
        }
        self.predict(sample)
        self.reset()
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _observation(episode: dict[str, Any]) -> dict[str, np.ndarray]:
        images = list(episode["images"].values())
        if not images:
            raise ValueError("episode contains no camera images")
        observation: dict[str, np.ndarray] = {
            "observation.state": np.asarray(episode["state"], dtype=np.float32)
        }
        for index, image in enumerate(images[:3], start=1):
            observation[f"observation.images.camera{index}"] = np.asarray(image, dtype=np.uint8)
        return observation


_default_runner = SmolVLARunner()


def initialize(model_path: str) -> None:
    _default_runner.initialize(model_path)


def run_inference(model_path: str, episode: dict[str, Any]) -> np.ndarray:
    del model_path
    return _default_runner.predict(episode)


def reset_episode() -> None:
    _default_runner.reset()
