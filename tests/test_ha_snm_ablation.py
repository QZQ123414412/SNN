import unittest

from models import modelpool
from parity_anchor_ftbc import named_signed_layers
from scripts.experiments.run_ha_snm_ablation import (
    CONFIGS,
    FAMILIES,
    MODES,
    build_parser,
    configure_snm,
)


class HASNMAblationTest(unittest.TestCase):
    def test_nine_configs_cover_three_ftbc_families_and_modes(self):
        observed = {
            (config["family"], config["mode"]) for config in CONFIGS.values()
        }
        self.assertEqual(
            observed,
            {(family, mode) for family in FAMILIES for mode in MODES},
        )
        self.assertEqual(len(CONFIGS), 9)

    def test_configure_ha_snm_propagates_global_schedule(self):
        model = modelpool("resnet20_signed", "cifar100").eval()
        configure_snm(model, "ha", start=1.25, end=0.5, reference=8)
        for module in named_signed_layers(model).values():
            self.assertTrue(module.enable_signed)
            self.assertEqual(module.snm_mode, "horizon_annealed")
            self.assertEqual(module.ha_snm_start, 1.25)
            self.assertEqual(module.ha_snm_end, 0.5)
            self.assertEqual(module.ha_snm_reference, 8.0)

    def test_off_and_standard_keep_legacy_mode(self):
        model = modelpool("resnet20_signed", "cifar100").eval()
        configure_snm(model, "off", start=1.25, end=0.5, reference=8)
        self.assertTrue(
            all(
                not module.enable_signed and module.snm_mode == "standard"
                for module in named_signed_layers(model).values()
            )
        )
        configure_snm(model, "standard", start=1.25, end=0.5, reference=8)
        self.assertTrue(
            all(
                module.enable_signed and module.snm_mode == "standard"
                for module in named_signed_layers(model).values()
            )
        )

    def test_parser_freezes_selected_defaults(self):
        args = build_parser().parse_args(
            ["--dataset", "cifar100", "--architecture", "resnet20", "-L", "8"]
        )
        self.assertEqual(args.ha_start, 1.25)
        self.assertEqual(args.ha_end, 0.5)
        self.assertEqual(args.ha_reference, 8.0)


if __name__ == "__main__":
    unittest.main()
