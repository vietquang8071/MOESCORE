#!/usr/bin/env python3
"""Install notebook requirements while preserving the existing GPU stack.

Standard-library only: this script runs before the project's dependencies are
imported. No installation takes place merely by importing this module.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys

TORCH_STACK = {"torch", "torchaudio", "torchvision"}


def normalized_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def protected(name):
    name = normalized_name(name)
    return name in TORCH_STACK | {"triton"} or name.startswith("nvidia-")


def installed_versions():
    return {normalized_name(dist.metadata["Name"]): dist.version
            for dist in metadata.distributions() if dist.metadata.get("Name")}


def write_runtime_constraints(path, versions):
    missing = TORCH_STACK - set(versions)
    if missing:
        raise RuntimeError(f"Missing preinstalled GPU-stack packages: {sorted(missing)}. "
                           "Use a Kaggle image with matching PyTorch/TorchAudio/TorchVision first.")
    frozen = {name: version for name, version in versions.items() if protected(name)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Preserve this runtime's GPU stack\n" + "".join(
        f"{name}=={frozen[name]}\n" for name in sorted(frozen)), encoding="utf-8")
    return frozen


def build_install_command(requirements, constraints, *, wheelhouse=None, extras=(), install_requirements=True):
    command = [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
               "--constraint", str(constraints)]
    if install_requirements:
        if not Path(requirements).is_file():
            raise FileNotFoundError(f"Requirements file is missing: {requirements}")
        command += ["--requirement", str(requirements)]
    for spec in extras:
        match = re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*", spec)
        if not match or protected(match.group()):
            raise ValueError(f"EXTRA_PACKAGES cannot replace the GPU stack or contain pip flags: {spec!r}")
    if not install_requirements and not extras:
        raise ValueError("No requirements or extra packages requested")
    if wheelhouse is not None:
        if not Path(wheelhouse).is_dir():
            raise NotADirectoryError(f"Offline wheel directory is missing: {wheelhouse}")
        command += ["--no-index", "--find-links", str(wheelhouse)]
    return command + list(extras)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", default=str(Path(__file__).resolve().parents[1] / "requirements-kaggle.txt"))
    parser.add_argument("--constraints", required=True)
    parser.add_argument("--wheelhouse")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--no-requirements", action="store_true")
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)
    before = installed_versions()
    frozen = write_runtime_constraints(args.constraints, before)
    command = build_install_command(args.requirements, args.constraints,
        wheelhouse=args.wheelhouse, extras=args.extra, install_requirements=not args.no_requirements)
    subprocess.run(command, check=True)
    after = installed_versions()
    if {name: after.get(name) for name in frozen} != frozen:
        raise RuntimeError("The preinstalled GPU stack changed during installation; stop and restore the runtime")
    report = {"status": "PASS", "gpu_stack_preserved": frozen,
              "requirements": None if args.no_requirements else str(Path(args.requirements).resolve()),
              "changed_distributions": {name: {"before": before.get(name), "after": value}
                                        for name, value in after.items() if before.get(name) != value},
              "offline_wheelhouse": args.wheelhouse}
    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
