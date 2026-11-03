import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERT_IMPORTS = {
    "expert1": (
        "expert1_finetune.Train_Expert1",
        ("Expert1", "CLAPAudioTextEncoder"),
    ),
    "expert2": (
        "expert2_finetune.expert2_model",
        ("Expert2", "create_expert2"),
    ),
    "expert3": (
        "expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3",
        ("MoescoreExpert3",),
    ),
    "expert4": ("expert4_finetune.model", ("Expert4",)),
}


IMPORT_SCRIPT = r'''
import importlib
import inspect
import os
from pathlib import Path
import socket
import sys
import traceback

repo = Path.cwd().resolve()
imports = __IMPORTS__
def _deny_network(*args, **kwargs):
    raise RuntimeError("network disabled by LOCAL-03 import test")

socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
socket.create_connection = _deny_network
try:
    socket.create_connection(("127.0.0.1", 9), timeout=0.1)
except RuntimeError as exc:
    if "network disabled by LOCAL-03 import test" not in str(exc):
        raise
    print("NETWORK_BLOCKED")
else:
    raise AssertionError("import test network guard did not block a connection")

module_paths = {
    "expert1": repo / "expert1_finetune",
    "expert2": repo / "expert2_finetune",
    "expert3": repo / "expert3_finetune",
    "expert4": repo / "expert4_finetune",
}
before_sys_path = list(sys.path)
previous_sys_path = list(sys.path)
before_cwd = Path.cwd().resolve()
path_changes = []

try:
    for expert_name in __ORDER__:
        module_name, symbols = imports[expert_name]
        module = importlib.import_module(module_name)
        if sys.path != previous_sys_path:
            path_changes.append(
                {
                    "after": module_name,
                    "added": [entry for entry in sys.path if entry not in previous_sys_path],
                    "removed": [entry for entry in previous_sys_path if entry not in sys.path],
                }
            )
            previous_sys_path = list(sys.path)
        module_file = Path(module.__file__).resolve()
        if not module_file.is_relative_to(module_paths[expert_name]):
            raise AssertionError(
                f"{expert_name} module resolved outside expected source tree: {module_file}"
            )
        for symbol in symbols:
            value = getattr(module, symbol)
            symbol_file = Path(inspect.getfile(value)).resolve()
            if not symbol_file.is_relative_to(module_paths[expert_name]):
                raise AssertionError(
                    f"{expert_name}.{symbol} resolved to unexpected source: {symbol_file}"
                )
            print(f"IMPORTED {expert_name}.{symbol} from {symbol_file}")

    __PATH_MUTATION__

    vendor_roots = {
        "models": repo / "expert2_finetune" / "MGA-CLAP",
        "tools": repo / "expert2_finetune" / "MGA-CLAP",
        "BEATs": repo / "expert4_finetune" / "unilm" / "beats",
    }
    for module_name, expected_root in vendor_roots.items():
        vendor_module = sys.modules.get(module_name)
        if vendor_module is None:
            print(
                f"NOT_EXERCISED {module_name}: the expert module-level import path "
                "does not call its model factory"
            )
            continue

        origins = []
        module_file = getattr(vendor_module, "__file__", None)
        if module_file:
            origins.append(Path(module_file).resolve())
        module_spec = getattr(vendor_module, "__spec__", None)
        spec_origin = getattr(module_spec, "origin", None)
        if spec_origin and spec_origin not in {"built-in", "frozen"}:
            origins.append(Path(spec_origin).resolve())
        for package_path in getattr(vendor_module, "__path__", ()):
            origins.append(Path(package_path).resolve())
        origins = list(dict.fromkeys(origins))
        if not origins:
            raise AssertionError(
                f"{module_name} is loaded but has no inspectable __file__, __path__, or spec origin"
            )
        unexpected = [
            origin for origin in origins if not origin.is_relative_to(expected_root.resolve())
        ]
        if unexpected:
            raise AssertionError(
                f"vendor namespace {module_name} resolved outside {expected_root}: "
                f"origins={origins}, unexpected={unexpected}"
            )
        print(f"VENDOR_ORIGIN {module_name} {origins}")

    added_paths = [entry for entry in sys.path if entry not in before_sys_path]
    removed_paths = [entry for entry in before_sys_path if entry not in sys.path]
    if sys.path != before_sys_path:
        laion_module = sys.modules.get("laion_clap")
        laion_file = getattr(laion_module, "__file__", None)
        laion_root = Path(laion_file).resolve().parent if laion_file else None
        unexpected_added = [
            entry
            for entry in added_paths
            if laion_root is None or Path(entry).resolve() != laion_root
        ]
        normalized_sys_path = list(sys.path)
        for entry in added_paths:
            if entry in normalized_sys_path:
                normalized_sys_path.remove(entry)
        if unexpected_added or removed_paths or normalized_sys_path != before_sys_path:
            raise AssertionError(
                "expert module imports made unexpected sys.path changes or reorder; "
                f"per-import changes={path_changes}, unexpected_added={unexpected_added}, "
                f"removed={removed_paths}, baseline_order_preserved="
                f"{normalized_sys_path == before_sys_path}"
            )
        print(
            "EXTERNAL_SYS_PATH_ADDITION laion_clap "
            f"{[str(Path(entry).resolve()) for entry in added_paths]}"
        )
    else:
        print("NO_SYS_PATH_CHANGE")
    if Path.cwd().resolve() != before_cwd:
        raise AssertionError("expert module imports unexpectedly changed the current directory")
except Exception:
    traceback.print_exc()
    raise
'''


class ExpertImportIsolationTests(unittest.TestCase):
    def build_import_script(self, order, path_mutation="pass"):
        imports = {
            name: (module, symbols) for name, (module, symbols) in EXPERT_IMPORTS.items()
        }
        return (
            IMPORT_SCRIPT.replace("__IMPORTS__", repr(imports))
            .replace("__ORDER__", repr(order))
            .replace("__PATH_MUTATION__", path_mutation)
        )

    def run_import_order(self, order, *, path_mutation="pass", expect_success=True):
        script = self.build_import_script(order, path_mutation=path_mutation)
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=150,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                f"import order {order} timed out after {exc.timeout}s; "
                f"stdout={exc.stdout!r}; stderr={exc.stderr!r}"
            )
        if not expect_success:
            self.assertNotEqual(
                result.returncode,
                0,
                f"import order {order} unexpectedly accepted path mutation\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("sys.path", result.stderr)
            return result.stdout
        self.assertEqual(
            result.returncode,
            0,
            f"import order {order} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        for expert_name in order:
            self.assertIn(f"IMPORTED {expert_name}.", result.stdout)
        self.assertIn("NETWORK_BLOCKED", result.stdout)
        for vendor_name in ("models", "tools", "BEATs"):
            self.assertTrue(
                f"NOT_EXERCISED {vendor_name}" in result.stdout
                or f"VENDOR_ORIGIN {vendor_name}" in result.stdout,
                f"missing origin or NOT_EXERCISED diagnostic for {vendor_name}:\n{result.stdout}",
            )
        for line in result.stdout.splitlines():
            if line.startswith(("IMPORTED ", "VENDOR_ORIGIN ", "NOT_EXERCISED ", "EXTERNAL_SYS_PATH_ADDITION ", "NO_SYS_PATH_CHANGE", "NETWORK_BLOCKED")):
                print(f"{order}: {line}")
        return result.stdout

    def test_expert2_then_expert4(self):
        self.run_import_order(["expert2", "expert4"])

    def test_expert4_then_expert2(self):
        self.run_import_order(["expert4", "expert2"])

    def test_expert1_to_expert4(self):
        self.run_import_order(["expert1", "expert2", "expert3", "expert4"])

    def test_expert4_to_expert1(self):
        self.run_import_order(["expert4", "expert3", "expert2", "expert1"])

    def test_import_checker_rejects_sys_path_reordering(self):
        self.run_import_order(
            ["expert2", "expert4"],
            path_mutation="sys.path.reverse()",
            expect_success=False,
        )

    def test_adapter_package_import_has_no_heavy_expert_side_effects(self):
        script = r'''
import os
from pathlib import Path
import sys
import torch

before_path = list(sys.path)
before_cwd = Path.cwd().resolve()
before_cuda = torch.cuda.is_initialized()
before_modules = set(sys.modules)

import model.expert_adapters as adapters
from model.expert_adapters.base import ExpertAdapter

assert adapters.ExpertAdapter is ExpertAdapter
assert sys.path == before_path, "adapter package changed sys.path"
assert Path.cwd().resolve() == before_cwd, "adapter package changed cwd"
assert torch.cuda.is_initialized() == before_cuda == False, "adapter import initialized CUDA"

new_modules = set(sys.modules) - before_modules
forbidden_roots = {"laion_clap", "models", "tools", "BEATs", "transformers"}
unexpected = sorted(name for name in new_modules if name.split(".", 1)[0] in forbidden_roots)
assert not unexpected, f"adapter import loaded expert/vendor modules: {unexpected}"
print("ADAPTER_IMPORT_SIDE_EFFECTS_OK")
'''
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"adapter import side-effect check failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("ADAPTER_IMPORT_SIDE_EFFECTS_OK", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
