"""Configuration loading and path helpers."""

from pathlib import Path

import yaml

# Repository root = two levels above src/lfa/
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path=None):
    """Load the YAML config. Defaults to ``configs/config.yaml``."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(cfg, key):
    """Return ``cfg['paths'][key]`` as an absolute Path (relative to repo root)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def image_search_dirs(cfg):
    """Folders searched for patient-CSV images: ``patient_images_dir`` then ``extra_image_dirs``."""
    dirs = [resolve_path(cfg, "patient_images_dir")]
    for d in cfg["paths"].get("extra_image_dirs") or []:
        p = Path(d)
        dirs.append(p if p.is_absolute() else REPO_ROOT / p)
    return dirs


def labeled_image_dirs(cfg):
    """``{folder: label}`` for folders whose images are all analysed (e.g. controls)."""
    out = {}
    for d, label in (cfg["paths"].get("labeled_image_dirs") or {}).items():
        p = Path(d)
        out[p if p.is_absolute() else REPO_ROOT / p] = label
    return out


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level="INFO"):
    import logging

    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s | %(message)s")
