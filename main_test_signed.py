# 测试SignedIF模型并输出脉冲统计
import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from models import modelpool
from preprocess import datapool
from utils import train, val, seed_all, get_logger
from models.layer import *
from models import SignedIF
from spike_stats import (
    collect_signed_spike_stats,
    format_spike_stats_report,
    reset_signed_spike_stats,
)

parser = argparse.ArgumentParser(description='PyTorch SNM Signed Spike Testing')
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg16_signed',type=str,help='model (use vgg16_signed for SNM)')
parser.add_argument('-id', '--identifier', type=str,help='model statedict identifier')

# test configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-T', '--time', default=16, type=int, help='snn simulation time')
parser.add_argument('--thresh', default=1.0, type=float, help='threshold for signed spike neurons')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    seed_all(args.seed)
    # preparing data
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    # preparing model
    model = modelpool(args.model, args.dataset)

    model_dir = '%s-checkpoints'% (args.dataset)
    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
    
    # if old version state_dict - convert IF thresholds to SignedIF
    keys = list(state_dict.keys())
    new_state_dict = {}
    #加上负阈值参数
    for k, v in state_dict.items():
        if "relu.up" in k:
            # Convert to thresh
            new_key = k[:-7]+'thresh'
            new_state_dict[new_key] = v
            # Add neg_thresh
            new_state_dict[k[:-7]+'neg_thresh'] = -v
        elif "up" in k:
            # Convert to thresh  
            new_key = k[:-2]+'thresh'
            new_state_dict[new_key] = v
            # Add neg_thresh
            new_state_dict[k[:-2]+'neg_thresh'] = -v
        elif "thresh" in k:
            # Already a thresh parameter, copy it and create neg_thresh
            new_state_dict[k] = v
            # Add neg_thresh
            new_state_dict[k[:-6]+'neg_thresh'] = -v
        else:
            # Other parameters (weights, biases, etc.)
            new_state_dict[k] = v

    # Load state dict with strict=False to allow missing keys
    model.load_state_dict(new_state_dict, strict=False)
    
    # Print loaded thresholds for debugging
    print("\nLoaded thresholds:")
    for name, module in model.named_modules():
        if isinstance(module, SignedIF):
            print(f"  {name}: pos_thresh={module.thresh.data.item():.4f}, neg_thresh={module.neg_thresh.data.item():.4f}")
            break  # Just print first one as example

    model.to(device)

    # Set time steps for signed spike neurons
    model.set_T(args.time)
    
    # Print thresholds after set_T
    print("\nThresholds after set_T:")
    for name, module in model.named_modules():
        if isinstance(module, SignedIF):
            print(f"  {name}: pos_thresh={module.thresh.data.item():.4f}, neg_thresh={module.neg_thresh.data.item():.4f}")
            break
    
    # Only set threshold if explicitly requested (not default)
    # Otherwise use the threshold loaded from checkpoint
    if args.thresh != 1.0 and hasattr(model, 'set_thresh'):
        model.set_thresh(args.thresh)
        print(f"Overriding thresholds to: {args.thresh}")

    print(f"\nTesting with Signed Spike + Memory neurons")
    print(f"Time steps: {args.time}")

    # Test
    reset_signed_spike_stats(model, SignedIF)
    acc = val(model, test_loader, device, args.time)
    print(f"Test Accuracy: {acc:.2f}%")
    layer_stats = collect_signed_spike_stats(model, SignedIF, nn.Conv2d, nn.Linear)
    print(format_spike_stats_report(layer_stats))



if __name__ == "__main__":
    main()

