import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('world_head', ROOT / 'prismatic/models/fast_lewm.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class CoreTests(unittest.TestCase):
    def test_blocks_and_gradient(self):
        x = torch.randn(2, 50, 14, requires_grad=True)
        y = core.scale_action_conditioning_gradient(x, gradient_scale=0.1)
        self.assertEqual(core.build_action_blocks(y, num_prefixes=5).shape, (2, 5, 140))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 0.1))

    def test_prefix_causality(self):
        torch.manual_seed(1)
        encoder = core.ActionPrefixEncoder(state_dim=4, action_block_dim=6,
            prefix_dim=12, depth=1, num_heads=3, dropout=0).eval()
        state, blocks = torch.randn(2, 4), torch.randn(2, 5, 6)
        changed = blocks.clone()
        changed[:, 3:] += 10
        torch.testing.assert_close(encoder(state, blocks)[:, :3],
                                   encoder(state, changed)[:, :3])

    def test_transformer_backward(self):
        torch.manual_seed(2)
        head = core.PrefixConditionedWindowTransformerHead(query_dim=6,
            target_dim=4, prefix_dim=8, model_dim=12, depth=2, num_heads=3,
            mlp_dim=24, window_sizes=(2,), num_horizons=5, dropout=0)
        query = torch.randn(1, 3, 4, 4, 6, requires_grad=True)
        prefix = torch.randn(1, 5, 8, requires_grad=True)
        pred = head(query, prefix)
        self.assertEqual(pred.shape, (1, 3, 5, 4, 4, 4))
        core.horizon_weighted_cosine_loss(pred, torch.randn_like(pred),
            horizon_weights=(1, 1, 1, 1, 1)).backward()
        for grad in (query.grad, prefix.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
