"""Import-only checks for the root MoE training pipeline."""

import importlib
import unittest


REQUIRED_IMPORTS = {
    "data.dataset": ("get_dataset",),
    "loss.loss_function": ("get_loss_function",),
    "model.wrapper": ("TextEncoder", "AudioEncoder"),
    "model.Basic_comp": ("Expert1", "Expert2", "Expert3", "Expert4"),
    "model.MOE_model": ("MOEModel",),
    "Train_MOE": (),
}


class ImportSmokeTests(unittest.TestCase):
    def test_required_modules_import_without_initializing_models(self):
        failures = []
        for module_name, symbols in REQUIRED_IMPORTS.items():
            try:
                module = importlib.import_module(module_name)
                for symbol in symbols:
                    getattr(module, symbol)
            except Exception as exc:  # preserve the original import exception
                failures.append(
                    f"{module_name}: {type(exc).__name__}: {exc}"
                )
            else:
                print(f"PASS {module_name}")

        self.assertFalse(failures, "Import failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
