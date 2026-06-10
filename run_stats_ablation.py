# 运行四配置脉冲统计消融实验
"""
Ablation study for spike statistics using the thesis configurations:

  A_QCFS
  B_QCFS+SNM
  C_QCFS+SNM+R0
  D_QCFS+FTBC
  E_QCFS+SNM+FTBC
  F_QCFS+SNM+R0+FTBC

Reported metrics:
  - input-driven SOPs
  - positive / negative spike rate
  - per-layer spike sparsity
"""

import argparse
import copy
import os
import time

import torch
import torch.nn as nn

from calibration import bias_corr_model
from models import SignedIF, modelpool
from preprocess import datapool
from spike_stats import collect_signed_spike_stats, reset_signed_spike_stats
from utils import seed_all, val


CONFIGS = {
    "A_QCFS": dict(signed=False, r0=False, ftbc=False),
    "B_QCFS+SNM": dict(signed=True, r0=False, ftbc=False),
    "C_QCFS+SNM+R0": dict(signed=True, r0=True, ftbc=False),
    "D_QCFS+FTBC": dict(signed=False, r0=False, ftbc=True),
    "E_QCFS+SNM+FTBC": dict(signed=True, r0=False, ftbc=True),
    "F_QCFS+SNM+R0+FTBC": dict(signed=True, r0=True, ftbc=True),
}


def load_signed_model(args, device):
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


def summarize_layer_stats(layer_stats):
    total_pos = sum(item.positive_spikes for item in layer_stats)
    total_neg = sum(item.negative_spikes for item in layer_stats)
    total_obs = sum(item.total_observations for item in layer_stats)
    total_sops = sum(item.sops for item in layer_stats)
    total_spikes = total_pos + total_neg
    return {
        "positive_spikes": total_pos,
        "negative_spikes": total_neg,
        "positive_rate": total_pos / max(total_obs, 1),
        "negative_rate": total_neg / max(total_obs, 1),
        "total_rate": total_spikes / max(total_obs, 1),
        "sparsity": 1.0 - total_spikes / max(total_obs, 1),
        "sops": total_sops,
    }


def run_one(cfg, model_template, ann_template, T, train_loader, test_loader, device, args):
    model = copy.deepcopy(model_template)
    model.to(device)
    model.set_signed(cfg["signed"])
    model.set_r0(cfg["r0"])
    model.reset_all_bias()
    model.set_T(T)

    if cfg["ftbc"]:
        ann = copy.deepcopy(ann_template)
        ann.to(device)
        ann.set_signed(False)
        ann.set_r0(False)
        bias_corr_model(
            ann=ann,
            snn=model,
            T=T,
            train_loader=train_loader,
            curr_t_alpha=args.alpha,
            num_cali_sample_batches=args.cali_batches,
        )
        del ann
        torch.cuda.empty_cache()

    reset_signed_spike_stats(model, SignedIF)
    acc = val(model, test_loader, device, T)
    layer_stats = collect_signed_spike_stats(model, SignedIF, nn.Conv2d, nn.Linear)
    summary = summarize_layer_stats(layer_stats)

    del model
    torch.cuda.empty_cache()
    return acc, summary, layer_stats


def format_pct(value):
    return f"{value * 100:.4f}%"


def write_markdown_report(path, args, results, layer_results):
    Ts = args.time_steps
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Spike Statistics Ablation on {args.dataset.upper()} / VGG16\n\n")
        f.write(f"- Configs: {', '.join(CONFIGS.keys())}\n")
        f.write(f"- Calibration: alpha={args.alpha}, batches={args.cali_batches}\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write("- SOPs convention: input-driven SOPs. The raw image input before the first spiking layer is not counted as a spike source.\n\n")

        metric_specs = [
            ("Accuracy", "acc", lambda v: f"{v:.2f}%"),
            ("Input-driven SOPs", "sops", lambda v: f"{v:,}"),
            ("Positive spike rate", "positive_rate", format_pct),
            ("Negative spike rate", "negative_rate", format_pct),
            ("Overall spike sparsity", "sparsity", format_pct),
            ("Elapsed", "elapsed", lambda v: f"{v:.1f}s"),
        ]

        for title, key, formatter in metric_specs:
            f.write(f"## {title}\n\n")
            f.write("| Config | " + " | ".join(f"T={t}" for t in Ts) + " |\n")
            f.write("|" + "---|" * (len(Ts) + 1) + "\n")
            for cfg_name in CONFIGS:
                cells = [
                    formatter(results[cfg_name][t][key]) if t in results[cfg_name] else "-"
                    for t in Ts
                ]
                f.write(f"| {cfg_name} | " + " | ".join(cells) + " |\n")
            f.write("\n")

        f.write("## Per-layer Detail\n\n")
        for cfg_name in CONFIGS:
            for T in Ts:
                if T not in layer_results[cfg_name]:
                    continue
                f.write(f"### {cfg_name}, T={T}\n\n")
                f.write("| Layer | PosRate | NegRate | Sparsity | InputSpikes | SOPs |\n")
                f.write("|---|---:|---:|---:|---:|---:|\n")
                for item in layer_results[cfg_name][T]:
                    f.write(
                        f"| {item.name} | {format_pct(item.positive_spike_rate)} | "
                        f"{format_pct(item.negative_spike_rate)} | {format_pct(item.spike_sparsity)} | "
                        f"{item.total_input_spikes:,} | {item.sops:,} |\n"
                    )
                f.write("\n")


def print_summary_table(results, Ts):
    header = f"{'Config':<24} {'T':>4} {'Acc':>8} {'SOPs':>18} {'PosRate':>10} {'NegRate':>10} {'Sparsity':>10} {'Elapsed':>9}"
    print("\n" + header)
    print("-" * len(header))
    for cfg_name in CONFIGS:
        for T in Ts:
            row = results[cfg_name][T]
            print(
                f"{cfg_name:<24} {T:>4} {row['acc']:>7.2f}% "
                f"{row['sops']:>18,d} {format_pct(row['positive_rate']):>10} "
                f"{format_pct(row['negative_rate']):>10} {format_pct(row['sparsity']):>10} "
                f"{row['elapsed']:>8.1f}s"
            )


def main():
    parser = argparse.ArgumentParser(description="Spike statistics ablation")
    parser.add_argument("-data", "--dataset", default="cifar100", type=str)
    parser.add_argument("-id", "--identifier", type=str, required=True)
    parser.add_argument("-dev", "--device", default="0", type=str)
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--cali_batches", default=5, type=int)
    parser.add_argument("--time_steps", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--output", default="", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    model_template = load_signed_model(args, device)
    ann_template = copy.deepcopy(model_template)

    results = {cfg_name: {} for cfg_name in CONFIGS}
    layer_results = {cfg_name: {} for cfg_name in CONFIGS}

    for cfg_name, cfg in CONFIGS.items():
        for T in args.time_steps:
            tag = f"{cfg_name} T={T}"
            print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}")
            start = time.time()
            acc, summary, layer_stats = run_one(
                cfg, model_template, ann_template, T, train_loader, test_loader, device, args
            )
            elapsed = time.time() - start
            summary["acc"] = acc
            summary["elapsed"] = elapsed
            results[cfg_name][T] = summary
            layer_results[cfg_name][T] = layer_stats
            print(
                f"{tag}: acc={acc:.2f}% sops={summary['sops']:,} "
                f"pos={format_pct(summary['positive_rate'])} "
                f"neg={format_pct(summary['negative_rate'])} "
                f"sparsity={format_pct(summary['sparsity'])} "
                f"elapsed={elapsed:.1f}s"
            )
            if args.output:
                write_markdown_report(args.output, args, results, layer_results)

    print_summary_table(results, args.time_steps)
    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        f"STATS_ABLATION_{args.dataset}.md",
    )
    write_markdown_report(output_path, args, results, layer_results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
