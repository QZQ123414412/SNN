import argparse
import time

import torch

from models import modelpool


def build_model(name, device):
    model_name = "vgg16_signed" if name == "vgg16" else "resnet34"
    model = modelpool(model_name, "imagenet").to(device).eval()
    if hasattr(model, "set_signed"):
        model.set_signed(True)
    if hasattr(model, "set_r0"):
        model.set_r0(True)
    if hasattr(model, "set_ftbc_mode"):
        model.set_ftbc_mode("none")
    return model


def benchmark(name, time_steps, batch_size, warmup, iterations, device):
    model = build_model(name, device)
    model.set_T(time_steps)
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs).mean(0)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(iterations):
            model(inputs).mean(0)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    seconds_per_batch = elapsed / iterations
    seconds_per_image = seconds_per_batch / batch_size
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    return seconds_per_image, peak_gib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--temporal_batch", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(device))
    print("model,T,batch,seconds_per_image,peak_GiB")
    for name in ("vgg16", "resnet34"):
        for time_steps in (4, 8, 16, 32):
            batch_size = max(args.temporal_batch // time_steps, 1)
            seconds, peak = benchmark(
                name,
                time_steps,
                batch_size,
                args.warmup,
                args.iterations,
                device,
            )
            print(
                f"{name},{time_steps},{batch_size},"
                f"{seconds:.6f},{peak:.3f}",
                flush=True,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
