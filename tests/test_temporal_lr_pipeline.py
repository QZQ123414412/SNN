import unittest

import torch

from models import SignedIF, modelpool
from spike_stats import summarize_ftbc_storage
from temporal_lr import (
    compress_full_ftbc_teacher,
    gate_groups,
    named_signed_layers,
    set_group_margins,
    snm_runtime_state_bytes_per_sample,
)
from scripts.experiments.run_temporal_lr_gated_snm import (
    make_deployment_compressed,
)


class TemporalLRPipelineTest(unittest.TestCase):
    def make_teacher(self, time_steps=4):
        model = modelpool("resnet20_signed", "cifar100").eval()
        model.set_T(time_steps)
        model.set_signed(False)
        model.set_ftbc_mode("full")
        with torch.no_grad():
            model(torch.zeros(2, 3, 32, 32))
        for layer_index, module in enumerate(named_signed_layers(model).values()):
            channels = module.time_based_bias[0].numel()
            channel_scale = torch.linspace(0.1, 1.0, channels)
            for t in range(time_steps):
                module.time_based_bias[t] = (
                    (layer_index + 1) * (t + 1) * channel_scale
                )
        return model

    def test_shared_temporal_basis_reconstructs_rank_one_teacher(self):
        model = self.make_teacher()
        teacher = {
            name: torch.stack(module.time_based_bias).clone()
            for name, module in named_signed_layers(model).items()
        }

        report = compress_full_ftbc_teacher(model, rank=2)

        self.assertGreater(report["explained_energy"], 0.999999)
        owners = 0
        shared_ids = set()
        for name, module in named_signed_layers(model).items():
            self.assertEqual(module.ftbc_mode, "temporal_low_rank")
            owners += int(module.owns_temporal_basis)
            shared_ids.add(id(module.temporal_basis))
            reconstructed = torch.matmul(
                module.temporal_basis,
                module.temporal_coeff,
            )
            self.assertTrue(torch.allclose(reconstructed, teacher[name], atol=1e-4))
        self.assertEqual(owners, 1)
        self.assertEqual(len(shared_ids), 1)

        storage = summarize_ftbc_storage(model, SignedIF)
        expected_coefficients = sum(
            module.temporal_coeff.numel()
            for module in named_signed_layers(model).values()
        )
        first = next(iter(named_signed_layers(model).values()))
        self.assertEqual(
            storage["parameters"],
            expected_coefficients + first.temporal_basis.numel(),
        )
        self.assertGreater(storage["synthesis_macs"], 0)

    def test_hybrid_keeps_selected_layer_full(self):
        model = self.make_teacher()
        final_name = "conv4_x.2.act"
        original = torch.stack(named_signed_layers(model)[final_name].time_based_bias)

        report = compress_full_ftbc_teacher(
            model,
            rank=2,
            full_layer_names=(final_name,),
        )

        final = named_signed_layers(model)[final_name]
        self.assertEqual(final.ftbc_mode, "full")
        self.assertTrue(torch.equal(torch.stack(final.time_based_bias), original))
        self.assertEqual(report["layers"][final_name]["representation"], "full")

    def test_rank_four_deployment_falls_back_to_full_at_t4(self):
        teacher = self.make_teacher(time_steps=4)
        model, report = make_deployment_compressed(
            teacher,
            rank=4,
            architecture="resnet20",
            time_steps=4,
        )
        self.assertTrue(report["fallback_to_full"])
        self.assertEqual(
            {module.ftbc_mode for module in named_signed_layers(model).values()},
            {"full"},
        )

    def test_residual_snm_margin_suppresses_small_correction(self):
        inputs = torch.tensor([[0.6], [-1.2]])

        standard = SignedIF(T=2, enable_signed=True, enable_r0=False).eval()
        standard.set_ftbc_mode("none")
        standard_output = standard(inputs).view(2, 1, 1)

        gated = SignedIF(T=2, enable_signed=True, enable_r0=False).eval()
        gated.set_ftbc_mode("none")
        gated.set_snm_negative_margin(0.5)
        gated_output = gated(inputs).view(2, 1, 1)

        self.assertEqual(float(standard_output[1].item()), -1.0)
        self.assertEqual(float(gated_output[1].item()), 0.0)

    def test_architecture_gate_groups_are_disjoint_and_assign_margins(self):
        model = modelpool("resnet20_signed", "cifar100")
        groups = gate_groups(model, "resnet20")
        self.assertEqual(list(groups), ["early", "middle", "late", "final"])
        margins = {"early": 0.0, "middle": 0.25, "late": 0.5, "final": 1.0}
        self.assertEqual(set_group_margins(model, "resnet20", margins), 4)
        layers = named_signed_layers(model)
        self.assertEqual(layers["conv1.2"].snm_negative_margin, 0.0)
        self.assertEqual(layers["conv3_x.0.act"].snm_negative_margin, 0.25)
        self.assertEqual(layers["conv4_x.0.act"].snm_negative_margin, 0.5)
        self.assertEqual(layers["conv4_x.2.act"].snm_negative_margin, 1.0)

    def test_runtime_state_storage_is_reported_per_sample(self):
        neuron = SignedIF(T=2).eval()
        neuron.set_ftbc_mode("none")
        neuron(torch.zeros(6, 4))
        holder = torch.nn.Sequential(neuron)
        expected = 2 * 4 * torch.tensor([], dtype=torch.float32).element_size()
        self.assertEqual(snm_runtime_state_bytes_per_sample(holder), expected)


if __name__ == "__main__":
    unittest.main()
