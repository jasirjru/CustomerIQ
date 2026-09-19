"""Print release blockers without evaluating data or changing release state."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.readiness import assess_from_files, report_as_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.v1.json"))
    parser.add_argument("--business-costs", type=Path, default=Path("docs/business-costs.json"))
    parser.add_argument("--evidence", type=Path, default=Path("docs/release-evidence.template.json"))
    args = parser.parse_args()
    report = assess_from_files(args.manifest, args.business_costs, args.evidence)
    print(report_as_json(report))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
