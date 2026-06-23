import unittest

import torch
import torch.nn as nn

from models.layer import IF
from scripts.train.refinement_finetune import (
    compute_dual_branch_loss,
    configure_trainable_stage,
    get_refinement_event_rate,
    sample_time_steps,
    set_refinement_proxy,
)


class TinyQCFS(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.BatchNorm1d(4),
            IF(L=4, thresh=1.0),
        )
        self.classifier = nn.Linear(4, 2, bias=False)

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class RefinementProxyActivationTest(unittest.TestCase):
    def test_proxy_preserves_forward_hard_events_and_backpropagates_to_threshold(self):
        activation = IF(L=4, thresh=1.0)
        activation.set_refinement_proxy(
            enabled=True,
            time_steps=4,
            schedule="custom",
            custom_weights=[0.5, 0.25, 0.125, 0.125],
            positive_margin=0.5,
            negative_margin=0.5,
        )
        activation.reset_refinement_proxy_stats()
        inputs = torch.tensor([[0.4, 0.9]], requires_grad=True)

        outputs = activation(inputs)
        event_rate = activation.get_refinement_event_rate()
        outputs.sum().backward()

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertGreater(event_rate.item(), 0.0)
        self.assertIsNotNone(activation.thresh.grad)
        self.assertGreater(activation.thresh.grad.abs().item(), 0.0)

    def test_time_step_one_proxy_returns_clean_qcfs_activation(self):
        activation = IF(L=4, thresh=1.0)
        inputs = torch.tensor([[0.37, 0.91]])
        clean = activation(inputs)

        activation.set_refinement_proxy(enabled=True, time_steps=1, schedule="binary")

        self.assertTrue(torch.equal(activation(inputs), clean))


class RefinementFinetuneUtilityTest(unittest.TestCase):
    def test_sample_time_steps_uses_supplied_probabilities(self):
        generator = torch.Generator().manual_seed(1)

        sampled = sample_time_steps(
            [2, 4, 8],
            probabilities=[0.0, 0.0, 1.0],
            generator=generator,
        )

        self.assertEqual(sampled, 8)

    def test_dual_branch_loss_returns_components_and_event_rate(self):
        model = TinyQCFS()
        inputs = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 0])

        loss, metrics = compute_dual_branch_loss(
            model,
            inputs,
            labels,
            time_steps=4,
            schedule="geometric",
            ratio=1.1,
            lambda_clean=0.5,
            lambda_cons=0.25,
            lambda_event=0.1,
        )
        loss.backward()

        self.assertIn("ce_refinement", metrics)
        self.assertIn("ce_clean", metrics)
        self.assertIn("kl_consistency", metrics)
        self.assertIn("event_rate", metrics)
        self.assertGreater(metrics["event_rate"].item(), 0.0)
        activation = next(module for module in model.modules() if isinstance(module, IF))
        self.assertIsNotNone(activation.thresh.grad)

    def test_stage_a_only_trains_thresholds_batchnorm_and_classifier(self):
        model = TinyQCFS()

        configure_trainable_stage(model, stage="A")

        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertIn("features.1.weight", trainable)
        self.assertIn("features.1.bias", trainable)
        self.assertIn("features.2.thresh", trainable)
        self.assertIn("classifier.weight", trainable)
        self.assertNotIn("features.0.weight", trainable)

    def test_stage_b_trains_entire_network(self):
        model = TinyQCFS()

        configure_trainable_stage(model, stage="B")

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_model_level_proxy_configuration_and_event_rate(self):
        model = TinyQCFS()
        set_refinement_proxy(
            model,
            enabled=True,
            time_steps=4,
            schedule="geometric",
            ratio=1.1,
        )
        _ = model(torch.randn(3, 4))

        self.assertGreater(get_refinement_event_rate(model).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
