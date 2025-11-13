"""Utilities that wrap Weights & Biases interactions behind a thin facade.

Keeping W&B usage in a single module makes it easier to mock during tests and
ensures the rest of the training codebase stays import-safe when wandb is not
installed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from treemort.utils.logger import get_logger

logger = get_logger(__name__)


def _require_wandb():
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "Weights & Biases support requested but the `wandb` package isn't installed."
        ) from exc
    return wandb


def _safe_conf_dict(conf) -> Dict[str, Any]:
    conf_dict: Dict[str, Any] = {}
    for key, value in vars(conf).items():
        try:
            conf_dict[key] = value
        except TypeError:
            conf_dict[key] = str(value)
    return conf_dict


def _git_info() -> Dict[str, str]:
    def _run(cmd: Iterable[str]) -> str:
        return subprocess.check_output(cmd, text=True).strip()

    info: Dict[str, str] = {}
    try:
        info["git_commit"] = _run(["git", "rev-parse", "HEAD"])
    except Exception:
        return info

    try:
        info["git_branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    return info


def init_wandb_run(conf, config_paths: Optional[list[str]] = None):
    if not getattr(conf, "wandb", False):
        return None

    wandb = _require_wandb()
    resume = "allow" if getattr(conf, "wandb_run_id", None) else None
    run = wandb.init(
        entity=getattr(conf, "wandb_entity", None) or None,
        project=getattr(conf, "wandb_project", None) or "treemort",
        config=_safe_conf_dict(conf),
        id=getattr(conf, "wandb_run_id", None) or None,
        resume=resume,
    )

    git_meta = _git_info()
    if git_meta:
        run.config.update(git_meta, allow_val_change=True)

    if config_paths:
        for cfg_path in config_paths:
            if cfg_path and os.path.exists(cfg_path):
                run.save(cfg_path)

    try:
        run.log_code()
    except Exception as exc:  # pragma: no cover - optional feature
        logger.warning(f"Failed to snapshot code in W&B: {exc}")

    return run


def finish_wandb_run(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # pragma: no cover - logging best effort
        logger.warning(f"Failed to close W&B run cleanly: {exc}")


def use_dataset_artifact(run, artifact_ref: str, download_root: Optional[str] = None) -> Dict[str, Any]:
    if not artifact_ref:
        return {}

    wandb = _require_wandb()
    artifact = None
    if run is not None:
        artifact = run.use_artifact(artifact_ref, type="dataset")
    else:
        artifact = wandb.use_artifact(artifact_ref, type="dataset")

    target_dir = artifact.download(root=download_root) if download_root else artifact.download()
    target_path = Path(target_dir)

    hdf5_files = list(target_path.rglob("*.h5"))
    hdf5_file = hdf5_files[0] if hdf5_files else None

    metadata = {
        "dataset_artifact_dir": str(target_path),
        "dataset_artifact_name": artifact.name,
        "dataset_artifact_version": artifact.version,
    }
    if hdf5_file is not None:
        metadata["dataset_hdf5_file"] = str(hdf5_file)

    return metadata


def init_weights_from_registry(model, artifact_ref: str, device, run=None) -> Optional[Path]:
    if not artifact_ref:
        return None

    wandb = _require_wandb()
    artifact = None
    if run is not None:
        artifact = run.use_artifact(artifact_ref, type="model")
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_ref, type="model")

    artifact_dir = Path(artifact.download())
    checkpoint_candidates = sorted(artifact_dir.rglob("*.pth"))
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No .pth checkpoint found inside artifact {artifact_ref} (downloaded to {artifact_dir})."
        )

    best_match = checkpoint_candidates[0]
    state_dict = torch.load(best_match, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model weights from W&B artifact: {artifact_ref} ({best_match}).")
    return best_match


def log_model_artifact(run, file_path: str, name: str, aliases: Iterable[str], metadata: Optional[Dict[str, Any]] = None):
    if run is None or not file_path or not os.path.exists(file_path):
        return None

    wandb = _require_wandb()
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
    artifact.add_file(file_path)
    logged_artifact = run.log_artifact(artifact, aliases=list(aliases))
    logger.info(f"Logged model artifact {name} with aliases {aliases}.")
    return logged_artifact


def format_aliases(run_id: str, *extra_aliases: str) -> list[str]:
    aliases = [alias for alias in extra_aliases if alias]
    if run_id:
        aliases.append(run_id)
    return aliases


def promote_artifact_alias(artifact_ref: str, alias: str, artifact_type: str = "model"):
    if not artifact_ref or not alias:
        raise ValueError("Both artifact_ref and alias are required")

    wandb = _require_wandb()
    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type=artifact_type)
    if alias not in artifact.aliases:
        artifact.aliases.append(alias)
        artifact.save()
        logger.info(f"Added alias '{alias}' to artifact {artifact_ref}.")
    else:
        logger.info(f"Artifact {artifact_ref} already has alias '{alias}'.")
    return artifact
