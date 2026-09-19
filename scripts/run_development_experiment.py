"""Generate reproducible development evidence without touching release data."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED
from src.data.make_dataset import load_development_partitions
from src.models.development_experiment import (
    DevelopmentExperimentConfig,
    run_development_experiment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_context() -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--calibration-outer-splits", type=int, default=5)
    parser.add_argument("--calibration-inner-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    X_train, X_validation, y_train, y_validation = load_development_partitions()
    config = DevelopmentExperimentConfig(
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        calibration_outer_splits=args.calibration_outer_splits,
        calibration_inner_splits=args.calibration_inner_splits,
        n_jobs=args.n_jobs,
    )
    evidence = run_development_experiment(
        X_train,
        X_validation,
        y_train,
        y_validation,
        config=config,
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_context(),
        "development_data_hashes": {
            name: _sha256(DATA_PROCESSED / name)
            for name in ("X_train_raw.csv", "y_train.csv")
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
        },
        "experiment": evidence,
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote development-only evidence to {output}")
    print(
        "Selected "
        f"{evidence['model_selection']['selected_candidate']} with calibration "
        f"method {evidence['calibrated_validation']['selected_method']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
