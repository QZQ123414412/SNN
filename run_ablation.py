"""
Ablation study: QCFS baseline  →  +SNM  →  +R0  →  +FTBC  (and combinations)
on CIFAR-100 / VGG16, sweeping T ∈ {2, 4, 8, 16, 32}.

Six configurations:
  A) QCFS          – positive-only spike, no R0, no FTBC
  B) QCFS+SNM      – signed spike + memory gate, no R0, no FTBC
  C) QCFS+SNM+R0   – signed spike + memory gate + R0, no FTBC
  D) QCFS+FTBC     – positive-only spike, no R0, WITH FTBC
  E) QCFS+SNM+FTBC – signed spike + memory gate, no R0, WITH FTBC
  F) QCFS+SNM+R0+FTBC – full SNM++ + FTBC  (our best)
"""

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn

from models import modelpool, SignedIF
from models.layer import IF
from preprocess import datapool
from utils import val, seed_all
from calibration import bias_corr_model

CONFIGS = {
    "A_QCFS":            dict(signed=False, r0=False, ftbc=False),
    "B_QCFS+SNM":        dict(signed=True,  r0=False, ftbc=False),
    "C_QCFS+SNM+R0":     dict(signed=True,  r0=True,  ftbc=False),
    "D_QCFS+FTBC":       dict(signed=False, r0=False, ftbc=True),
    "E_QCFS+SNM+FTBC":   dict(signed=True,  r0=False, ftbc=True),
    "F_QCFS+SNM+R0+FTBC": dict(signed=True, r0=True,  ftbc=True),
}


def load_signed_model(args, device):
    """Build VGG_Signed, load checkpoint, convert IF keys → SignedIF keys."""
    model = modelpool("vgg16_signed", args.dataset)
    ckpt = torch.load(
        os.path.join(f"{args.dataset}-checkpoints", args.identifier + ".pth"),
        map_location="cpu",
    )
    new_sd = {}
    for k, v in ckpt.items():
        if "relu.up" in k:
            base = k[:-7]
            new_sd[base + "thresh"] = v
            new_sd[base + "neg_thresh"] = -v
        elif "up" in k:
            base = k[:-2]
            new_sd[base + "thresh"] = v
            new_sd[base + "neg_thresh"] = -v
        elif "thresh" in k and "neg_thresh" not in k:
            new_sd[k] = v
            new_sd[k[:-6] + "neg_thresh"] = -v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    return model


def run_one(cfg_name, cfg, model_template, ann_template,
            T, train_loader, test_loader, device, args):
    """Evaluate one (config, T) combination and return accuracy."""
    model = copy.deepcopy(model_template)
    model.to(device)

    model.set_signed(cfg["signed"])
    model.set_r0(cfg["r0"])
    model.reset_all_bias()
    model.set_T(T)

    if cfg["ftbc"]:
        ann = copy.deepcopy(ann_template)
        ann.to(device)
        # ANN reference always stays in original unsigned mode (trained with [0,1] clip)
        ann.set_signed(False)
        ann.set_r0(False)
        bias_corr_model(
            ann=ann, snn=model, T=T,
            train_loader=train_loader,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=args.cali_batches,
        )
        del ann
        torch.cuda.empty_cache()

    acc = val(model, test_loader, device, T)
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    parser = argparse.ArgumentParser(description="SNM++ FTBC Ablation")
    parser.add_argument("-data", "--dataset", default="cifar100", type=str)
    parser.add_argument("-id", "--identifier", type=str, required=True)
    parser.add_argument("-dev", "--device", default="0", type=str)
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--alpha", default=0.4, type=float,
                        help="EMA alpha for FTBC calibration")
    parser.add_argument("--cali_batches", default=5, type=int,
                        help="number of calibration mini-batches")
    parser.add_argument("--time_steps", nargs="+", type=int,
                        default=[2, 4, 8, 16, 32],
                        help="list of T values to test")
    parser.add_argument("--output_suffix", default="",
                        help="suffix appended before .md, e.g. '_v2' → ABLATION_RESULTS_cifar100_v2.md")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    train_loader, test_loader = datapool(args.dataset, args.batch_size)

    model_template = load_signed_model(args, device)
    ann_template = copy.deepcopy(model_template)  # stays at T=0

    results = {}   # results[cfg_name][T] = acc

    for cfg_name, cfg in CONFIGS.items():
        results[cfg_name] = {}
        for T in args.time_steps:
            tag = f"[{cfg_name}  T={T}]"
            print(f"\n{'='*60}")
            print(f"  {tag}")
            print(f"{'='*60}")
            t0 = time.time()
            acc = run_one(cfg_name, cfg, model_template, ann_template,
                          T, train_loader, test_loader, device, args)
            elapsed = time.time() - t0
            results[cfg_name][T] = acc
            print(f"  {tag}  Accuracy = {acc:.2f}%  ({elapsed:.1f}s)")

    # ── pretty-print table ──
    Ts = args.time_steps
    header = f"{'Config':<25}" + "".join(f"{'T='+str(t):>10}" for t in Ts)
    sep = "-" * len(header)
    print(f"\n\n{'='*len(header)}")
    print(f"  ABLATION RESULTS  ({args.dataset.upper()} / VGG16)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for cfg_name in CONFIGS:
        row = f"{cfg_name:<25}"
        for t in Ts:
            a = results[cfg_name].get(t, float("nan"))
            row += f"{a:>9.2f}%"
        print(row)
    print(sep)

    # ── write markdown report ──
    md_path = os.path.join(os.path.dirname(__file__), f"ABLATION_RESULTS_{args.dataset}{args.output_suffix}.md")
    with open(md_path, "w") as f:
        f.write(f"# Ablation Study: SNM++ + FTBC on {args.dataset.upper()} / VGG16\n\n")
        f.write(f"- Calibration: alpha={args.alpha}, batches={args.cali_batches}\n")
        f.write(f"- Seed: {args.seed}\n\n")
        f.write("| Config | " + " | ".join(f"T={t}" for t in Ts) + " |\n")
        f.write("|" + "---|" * (len(Ts) + 1) + "\n")
        for cfg_name in CONFIGS:
            cells = []
            for t in Ts:
                a = results[cfg_name].get(t, float("nan"))
                cells.append(f"**{a:.2f}%**")
            f.write(f"| {cfg_name} | " + " | ".join(cells) + " |\n")

        f.write("\n## Config Legend\n\n")
        f.write("| Flag | Meaning |\n|---|---|\n")
        f.write("| QCFS | Baseline positive-only IF with v(0)=θ/2 |\n")
        f.write("| +SNM | Signed spike + memory gate (neg spike only after pos) |\n")
        f.write("| +R0 | No-debt rule: if m(t)=0, v(t)←max(v(t),0) |\n")
        f.write("| +FTBC | Forward Temporal Bias Correction (per-timestep calibration) |\n")

    print(f"\nResults saved to {md_path}")


if __name__ == "__main__":
    main()
