import unittest
import torch

from expert4_finetune.seq_coattn import SeqCoAttn, RegressionHead


class Expert4ComponentsTests(unittest.TestCase):
    def test_attention_and_head_shapes(self):
        torch.manual_seed(42)
        fusion = SeqCoAttn()
        audio = torch.randn(2, 4, 512)
        text = torch.randn(2, 3, 512)
        a_mask = torch.tensor([[False, False, False, True], [False] * 4])
        t_mask = torch.tensor([[False, False, True], [False] * 3])
        fused = fusion(audio, text, a_mask, t_mask)
        self.assertEqual(tuple(fused.shape), (2, 1024))
        score = RegressionHead()(fused)
        self.assertEqual(tuple(score.shape), (2,))
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue((score.abs() <= 1).all())
        score.sum().backward()
        self.assertIsNotNone(fusion.audio_to_text_attn.in_proj_weight.grad)
        self.assertIsNotNone(fusion.text_to_audio_attn.in_proj_weight.grad)

    def test_bad_masks_are_rejected(self):
        fusion = SeqCoAttn()
        with self.assertRaises(ValueError):
            fusion(torch.randn(1, 3, 512), torch.randn(1, 2, 512),
                   torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
