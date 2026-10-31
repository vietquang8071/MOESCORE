import unittest
import importlib.util
from types import SimpleNamespace
import torch
from torch import nn
from expert4_finetune.text_encoder import TextEncoder, verify_clap_projection_state, verify_clap_roberta_state, roberta_base_config


class FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768, model_type='roberta')
        self.embeddings = nn.Embedding(4, 768)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(768, 768) for _ in range(12)])

    def forward(self, input_ids, attention_mask, return_dict=True):
        x = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            x = layer(x)
        return SimpleNamespace(last_hidden_state=x)


class TextTests(unittest.TestCase):
    def test_checkpoint_projection_shapes_are_required(self):
        state = {'module.text_projection.0.weight': torch.zeros(512, 768),
                 'module.text_projection.2.weight': torch.zeros(512, 512)}
        verify_clap_projection_state(state)
        with self.assertRaises(ValueError):
            verify_clap_projection_state({})

    def test_roberta_base_embeddings_match_checkpoint(self):
        state = {
            'module.text_branch.embeddings.word_embeddings.weight': torch.empty(50265, 768, device='meta'),
            'module.text_branch.embeddings.position_embeddings.weight': torch.empty(514, 768, device='meta'),
            'module.text_branch.embeddings.token_type_embeddings.weight': torch.empty(1, 768, device='meta'),
        }
        verify_clap_roberta_state(state)
        state['module.text_branch.embeddings.position_embeddings.weight'] = torch.empty(512, 768, device='meta')
        with self.assertRaisesRegex(ValueError, 'position_embeddings'):
            verify_clap_roberta_state(state)

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Transformers is optional for lightweight tests')
    def test_local_roberta_config_uses_checkpoint_dimensions(self):
        config = roberta_base_config()
        self.assertEqual((config.vocab_size, config.hidden_size, config.max_position_embeddings,
                          config.type_vocab_size, config.num_hidden_layers), (50265, 768, 514, 1, 12))

    def test_token_projection_and_freezing(self):
        branch = FakeText()
        tokenizer = lambda captions, **kw: {'input_ids': torch.tensor([[0, 1, 2, 0]]), 'attention_mask': torch.tensor([[1, 1, 1, 0]])}
        encoder = TextEncoder(branch, nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 512)), tokenizer)
        output, mask = encoder(['example'])
        self.assertEqual(tuple(output.shape), (1, 3, 512))
        self.assertEqual(mask.tolist(), [[False, False, True]])
        output.sum().backward()
        self.assertIsNone(branch.encoder.layer[0].weight.grad)
        self.assertIsNotNone(branch.encoder.layer[-1].weight.grad)
        self.assertIsNotNone(encoder.projection[0].weight.grad)


if __name__ == '__main__':
    unittest.main()
