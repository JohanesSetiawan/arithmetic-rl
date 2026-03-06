from __future__ import annotations

import os
from typing import Any, Optional

import torch


def _to_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("WandB logging expects scalar tensors")
        return value.detach().item()
    return value


class WandbLogger:
    def __init__(
        self,
        enabled: bool,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        token: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.enabled = enabled
        self._wandb = None

        if not enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "WandB logging was requested, but the 'wandb' package is not installed."
            ) from exc

        if token:
            os.environ.setdefault("WANDB_API_KEY", token)
            wandb.login(key=token, relogin=True)

        self._wandb = wandb
        self._wandb.init(
            project=project or "arithmetic-rl",
            entity=entity,
            name=run_name,
            config=config,
        )
        self._wandb.define_metric("trainer/step")
        self._wandb.define_metric("*", step_metric="trainer/step")

    def log(self, step: int, metrics: dict[str, Any], prefix: Optional[str] = None) -> None:
        if not self.enabled or self._wandb is None:
            return

        payload: dict[str, Any] = {"trainer/step": step}
        for key, value in metrics.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            payload[metric_name] = _to_scalar(value)

        self._wandb.log(payload)

    def log_groups(self, step: int, groups: dict[str, dict[str, Any]]) -> None:
        if not self.enabled or self._wandb is None:
            return

        payload: dict[str, Any] = {"trainer/step": step}
        for group_name, metrics in groups.items():
            for key, value in metrics.items():
                payload[f"{group_name}/{key}"] = _to_scalar(value)

        self._wandb.log(payload)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()
