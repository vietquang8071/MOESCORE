import importlib.util
import unittest
from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parents[1] / "MGA-CLAP/models/text_encoder.py"


class InitPolicyTest(unittest.TestCase):
    def test_upstream_pretrain_keeps_pretrained_bert_default(self):
        module_spec = importlib.util.spec_from_file_location("mga_text_encoder_for_test", SOURCE)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        calls = []

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, name):
                return cls()

        class FakeConfig:
            @classmethod
            def from_pretrained(cls, name):
                calls.append("config")
                return cls()

        class FakeBert(torch.nn.Module):
            config_class = FakeConfig

            def __init__(self, config=None, add_pooling_layer=False):
                super().__init__()
                calls.append("random")

            @classmethod
            def from_pretrained(cls, name, add_pooling_layer=False):
                calls.append("pretrained")
                return cls.__new_from_pretrained()

            @classmethod
            def __new_from_pretrained(cls):
                instance = cls.__new__(cls)
                torch.nn.Module.__init__(instance)
                return instance

        module.MODELS["bert-base-uncased"] = (FakeBert, FakeTokenizer, 768)
        module.TextEncoder({"text_encoder_args": {"type": "bert-base-uncased", "freeze": False}})
        self.assertIn("pretrained", calls)
        self.assertNotIn("random", calls)


if __name__ == "__main__":
    unittest.main()
