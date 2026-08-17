"""DIST fine-tuning for the 68.78% CIFAR-100/ResNet20 QCFS checkpoint."""

import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch
import torch.nn as nn

from models import IF
from preprocess import datapool
from scripts.experiments.qcfs_checkpoint import checkpoint_sha256, load_qcfs_pair
from scripts.experiments.run_resnet20_qcfs_ablation import (
    FORMAL_MAX_T32_CONVERSION_GAP,
    validate_t32_conversion,
)
from scripts.train.finetune_resnet20_qcfs import (
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    save_first_target,
    target_metadata,
    validate_qcfs_model,
    verify_checkpoint,
)
from utils import seed_all, val


DEFAULT_SOURCE = (
    REPO_ROOT
    / "cifar100-checkpoints"
    / "resnet20_L[8]_paper_bs128_noaa_seed42_lr002_resumable.pth"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "cifar100-checkpoints"
    / "resnet20_qcfs_distill_finetune_68_78_to_69_94"
)
DEFAULT_TEACHER = (
    REPO_ROOT
    / "cifar100-checkpoints"
    / "teachers"
    / "cifar100_resnet56-f2eff4c8.pt"
)
TEACHER_URL = (
    "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/"
    "resnet/cifar100_resnet56-f2eff4c8.pt"
)
TARGET_FILENAME = "resnet20_cifar100_qcfs_L8_target_ge69_94.pth"
GLOBAL_BEST_FILENAME = "resnet20_cifar100_qcfs_L8_global_best.pth"
TRAJECTORIES = OrderedDict(
    [
        (
            "KD_FT_WLR1E4_TLR1E5",
            {"weight_lr": 1e-4, "threshold_lr": 1e-5},
        ),
        (
            "KD_FT_WLR5E5_TLR5E6",
            {"weight_lr": 5e-5, "threshold_lr": 5e-6},
        ),
    ]
)
STUDENT_MEAN = tuple(value / 255.0 for value in (129.3, 124.1, 112.4))
STUDENT_STD = tuple(value / 255.0 for value in (68.2, 65.4, 70.4))
TEACHER_MEAN = (0.5070, 0.4865, 0.4409)
TEACHER_STD = (0.2673, 0.2564, 0.2761)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_parser():
    parser = argparse.ArgumentParser(
        description="DIST fine-tune the 68.78% CIFAR-100/ResNet20 QCFS model"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--trajectory_names", nargs="+", default=list(TRAJECTORIES)
    )
    parser.add_argument(
        "--weight_learning_rates",
        type=float,
        nargs="+",
        default=[item["weight_lr"] for item in TRAJECTORIES.values()],
    )
    parser.add_argument(
        "--threshold_learning_rates",
        type=float,
        nargs="+",
        default=[item["threshold_lr"] for item in TRAJECTORIES.values()],
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dist_tau", type=float, default=4.0)
    parser.add_argument("--dist_beta", type=float, default=1.0)
    parser.add_argument("--dist_gamma", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dist_weight", type=float, default=2.0)
    parser.add_argument("--target_accuracy", type=float, default=69.94)
    parser.add_argument("--expected_source_accuracy", type=float, default=68.78)
    parser.add_argument("--source_accuracy_tolerance", type=float, default=0.005)
    parser.add_argument("--expected_teacher_accuracy", type=float, default=72.63)
    parser.add_argument("--teacher_accuracy_tolerance", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    lengths = {
        len(args.trajectory_names),
        len(args.weight_learning_rates),
        len(args.threshold_learning_rates),
    }
    if len(lengths) != 1:
        raise ValueError("trajectory names and learning-rate lists must match")
    if len(set(args.trajectory_names)) != len(args.trajectory_names):
        raise ValueError("trajectory names must be unique")
    rates = list(args.weight_learning_rates) + list(
        args.threshold_learning_rates
    )
    if any(rate <= 0 for rate in rates):
        raise ValueError("all learning rates must be positive")
    for name in (
        "dist_tau",
        "dist_beta",
        "dist_gamma",
        "ce_weight",
        "dist_weight",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")


def configure_numerics():
    # Match the CUDA numerical semantics used to train and select the 68.78%
    # source checkpoint. Disabling cuDNN TF32 changes 17/10,000 predictions
    # for that exact checkpoint (68.78% -> 68.61%).
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class TeacherBasicBlock(nn.Module):
    """State-dict-compatible block for chenyaofo/pytorch-cifar-models."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(outputs + identity)


class TeacherCifarResNet(nn.Module):
    """Minimal ResNet implementation matching the official teacher weights."""

    def __init__(self, layers=(9, 9, 9), num_classes=100):
        super().__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, layers[0])
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        layers = [
            TeacherBasicBlock(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes
        layers.extend(
            TeacherBasicBlock(self.inplanes, planes)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.avgpool(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        return self.fc(outputs)


class TeacherInputAdapter(nn.Module):
    """Map repository-normalized images to the teacher's native normalization."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        scale = [left / right for left, right in zip(STUDENT_STD, TEACHER_STD)]
        bias = [
            (left - right) / std
            for left, right, std in zip(STUDENT_MEAN, TEACHER_MEAN, TEACHER_STD)
        ]
        self.register_buffer("input_scale", torch.tensor(scale).view(1, 3, 1, 1))
        self.register_buffer("input_bias", torch.tensor(bias).view(1, 3, 1, 1))

    def forward(self, inputs):
        return self.model(inputs * self.input_scale + self.input_bias)


def build_teacher_model():
    return TeacherCifarResNet(layers=(9, 9, 9), num_classes=100)


def load_teacher_checkpoint(path, device):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        state_dict = torch.hub.load_state_dict_from_url(
            TEACHER_URL,
            model_dir=str(path.parent),
            file_name=path.name,
            check_hash=True,
            progress=True,
            map_location="cpu",
        )
    else:
        state_dict = torch.load(path, map_location="cpu")
    model = build_teacher_model()
    model.load_state_dict(state_dict, strict=True)
    model = TeacherInputAdapter(model)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(path),
        "sha256": checkpoint_sha256(path),
        "url": TEACHER_URL,
        "architecture": "cifar100_resnet56",
        "origin": "chenyaofo_pytorch_cifar_models_pretrained",
        "input_mean": list(TEACHER_MEAN),
        "input_std": list(TEACHER_STD),
    }


def cosine_similarity(left, right, eps=1e-8):
    return (left * right).sum(1) / (
        left.norm(dim=1) * right.norm(dim=1) + eps
    )


def pearson_correlation(left, right, eps=1e-8):
    return cosine_similarity(
        left - left.mean(1, keepdim=True),
        right - right.mean(1, keepdim=True),
        eps,
    )


def inter_class_relation(student, teacher):
    return 1 - pearson_correlation(student, teacher).mean()


def intra_class_relation(student, teacher):
    return inter_class_relation(student.transpose(0, 1), teacher.transpose(0, 1))


class DISTLoss(nn.Module):
    """Official DIST inter/intra-class relation loss."""

    def __init__(self, beta=1.0, gamma=1.0, tau=4.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, student_logits, teacher_logits):
        student = (student_logits / self.tau).softmax(dim=1)
        teacher = (teacher_logits / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(student, teacher)
        intra_loss = self.tau**2 * intra_class_relation(student, teacher)
        return self.beta * inter_loss + self.gamma * intra_loss


def load_student(source, device):
    ann, _, metadata = load_qcfs_pair(
        source, "cifar100", "resnet20", device
    )
    ann.set_L(8)
    ann.set_T(0)
    ann.set_qcfs_training_profile("fixed_repo")
    validate_qcfs_model(ann)
    return ann, metadata


def split_student_parameters(model):
    threshold_ids = {
        id(module.thresh) for module in model.modules() if isinstance(module, IF)
    }
    if len(threshold_ids) != 19:
        raise RuntimeError(f"Expected 19 QCFS thresholds, found {len(threshold_ids)}")
    weight_parameters = []
    threshold_parameters = []
    for parameter in model.parameters():
        target = (
            threshold_parameters
            if id(parameter) in threshold_ids
            else weight_parameters
        )
        target.append(parameter)
    if len(threshold_parameters) != 19 or not weight_parameters:
        raise RuntimeError("Student optimizer parameter partition is invalid")
    return weight_parameters, threshold_parameters


def experiment_config(args, source_sha256, teacher_metadata):
    return {
        "architecture": "resnet20",
        "augmentation_profile": "paper_era",
        "autoaugment": False,
        "batch_size": args.batch_size,
        "cutout_length": 16,
        "dataset": "cifar100",
        "dist_beta": args.dist_beta,
        "dist_gamma": args.dist_gamma,
        "dist_tau": args.dist_tau,
        "dist_weight": args.dist_weight,
        "epochs_per_trajectory": args.epochs,
        "expected_source_accuracy": args.expected_source_accuracy,
        "source_accuracy_tolerance": args.source_accuracy_tolerance,
        "expected_teacher_accuracy": args.expected_teacher_accuracy,
        "teacher_accuracy_tolerance": args.teacher_accuracy_tolerance,
        "teacher_validation_batch_size": 256,
        "loss": "cross_entropy_plus_DIST",
        "ce_weight": args.ce_weight,
        "momentum": 0.9,
        "cudnn_allow_tf32": True,
        "cuda_matmul_allow_tf32": False,
        "optimizer": "SGD",
        "qcfs_L": 8,
        "qcfs_training_profile": "fixed_repo",
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": 0.0,
        "seed": args.seed,
        "source_path": str(Path(args.source).resolve()),
        "source_sha256": source_sha256,
        "target_accuracy": args.target_accuracy,
        "max_t32_conversion_gap": FORMAL_MAX_T32_CONVERSION_GAP,
        "teacher": teacher_metadata,
        "test_samples": 10000,
        "student_input_mean": list(STUDENT_MEAN),
        "student_input_std": list(STUDENT_STD),
        "threshold_learning_rates": list(args.threshold_learning_rates),
        "trajectory_names": list(args.trajectory_names),
        "weight_decay": args.weight_decay,
        "weight_learning_rates": list(args.weight_learning_rates),
        "weight_origin": "official_architecture_distilled_qat",
        "workers": args.workers,
    }


def trajectory_paths(output_dir, name):
    directory = Path(output_dir) / name
    return {
        "directory": directory,
        "best": directory / f"{name}.best.pth",
        "history": directory / f"{name}.history.json",
        "state": directory / f"{name}.train_state.pth",
        "summary": directory / f"{name}.summary.json",
    }


def train_distill_epoch(
    student,
    teacher,
    loader,
    optimizer,
    ce_loss,
    dist_loss,
    device,
    ce_weight,
    dist_weight,
):
    student.train()
    teacher.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    total_ce = 0.0
    total_dist = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        ce_value = ce_loss(student_logits, labels)
        dist_value = dist_loss(student_logits, teacher_logits)
        loss = ce_weight * ce_value + dist_weight * dist_value
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total += batch_size
        correct += student_logits.argmax(1).eq(labels).sum().item()
        total_loss += float(loss.detach())
        total_ce += float(ce_value.detach())
        total_dist += float(dist_value.detach())
    return {
        "loss_sum": total_loss,
        "ce_loss_sum": total_ce,
        "dist_loss_sum": total_dist,
        "train_accuracy": 100.0 * correct / total,
    }


def load_resume_state(path, config, name, weight_lr, threshold_lr):
    state = torch.load(path, map_location="cpu")
    expected = {
        "experiment_config": config,
        "trajectory": name,
        "weight_lr": weight_lr,
        "threshold_lr": threshold_lr,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"Resume {key} mismatch for {name}")
    return state


def run_trajectory(
    args,
    config,
    name,
    weight_lr,
    threshold_lr,
    teacher,
    device,
):
    paths = trajectory_paths(args.output_dir, name)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    configure_numerics()
    train_loader, test_loader = datapool(
        "cifar100", args.batch_size, augmentation_profile="paper_era"
    )
    student, _ = load_student(args.source, device)
    weight_parameters, threshold_parameters = split_student_parameters(student)
    optimizer = torch.optim.SGD(
        [
            {"params": weight_parameters, "lr": weight_lr},
            {"params": threshold_parameters, "lr": threshold_lr},
        ],
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    ce_loss = nn.CrossEntropyLoss().to(device)
    dist_loss = DISTLoss(
        beta=args.dist_beta, gamma=args.dist_gamma, tau=args.dist_tau
    ).to(device)
    history = []
    best_accuracy = float("-inf")
    best_epoch = None
    start_epoch = 0

    if args.resume and paths["state"].is_file():
        state = load_resume_state(
            paths["state"], config, name, weight_lr, threshold_lr
        )
        student.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng_state"])
        history = state["history"]
        best_accuracy = state["best_accuracy"]
        best_epoch = state["best_epoch"]
        start_epoch = state["epoch"] + 1
        print(f"[{name}] resumed at epoch {start_epoch}", flush=True)
    elif any(paths[key].exists() for key in ("best", "history", "state", "summary")):
        raise FileExistsError(
            f"Refusing to overwrite outputs for {name}; use --resume"
        )

    for epoch in range(start_epoch, args.epochs):
        current_weight_lr = optimizer.param_groups[0]["lr"]
        current_threshold_lr = optimizer.param_groups[1]["lr"]
        metrics = train_distill_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            ce_loss,
            dist_loss,
            device,
            args.ce_weight,
            args.dist_weight,
        )
        scheduler.step()
        test_accuracy = val(student, test_loader, device, 0)
        record = dict(
            metrics,
            epoch=epoch,
            test_accuracy=test_accuracy,
            threshold_learning_rate=current_threshold_lr,
            weight_learning_rate=current_weight_lr,
        )
        history.append(record)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            atomic_torch_save(student.state_dict(), paths["best"])
        if test_accuracy + 1e-12 >= args.target_accuracy:
            if save_first_target(
                student,
                args.output_dir,
                config,
                name,
                epoch,
                test_accuracy,
            ):
                print(
                    f"TARGET SAVED: {test_accuracy:.3f}% from {name} epoch {epoch}",
                    flush=True,
                )
        print(
            f"[{name}] epoch={epoch + 1:03d}/{args.epochs} "
            f"wlr={current_weight_lr:.8g} tlr={current_threshold_lr:.8g} "
            f"train={metrics['train_accuracy']:.3f}% "
            f"test={test_accuracy:.3f}% best={best_accuracy:.3f}%",
            flush=True,
        )
        atomic_json_save(
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "experiment_config": config,
                "history": history,
                "status": "complete" if epoch + 1 == args.epochs else "incomplete",
                "threshold_lr": threshold_lr,
                "trajectory": name,
                "weight_lr": weight_lr,
            },
            paths["history"],
        )
        atomic_torch_save(
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "experiment_config": config,
                "history": history,
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
                "scheduler": scheduler.state_dict(),
                "threshold_lr": threshold_lr,
                "trajectory": name,
                "weight_lr": weight_lr,
            },
            paths["state"],
        )

    if best_epoch is None:
        raise RuntimeError(f"No best checkpoint for {name}")
    verified_accuracy, _ = verify_checkpoint(
        paths["best"], best_accuracy, test_loader, device
    )
    result = {
        "best_accuracy": best_accuracy,
        "best_checkpoint": str(paths["best"].resolve()),
        "best_checkpoint_sha256": checkpoint_sha256(paths["best"]),
        "best_epoch": best_epoch,
        "completed_at_utc": utc_now(),
        "passed_ann_gate": best_accuracy + 1e-12 >= args.target_accuracy,
        "threshold_lr": threshold_lr,
        "trajectory": name,
        "verified_accuracy": verified_accuracy,
        "weight_lr": weight_lr,
    }
    atomic_json_save(result, paths["summary"])
    del student, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def copy_global_best(result, output_dir, config, resume):
    path = Path(output_dir) / GLOBAL_BEST_FILENAME
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        if not resume or not (path.exists() and metadata_path.exists()):
            raise FileExistsError("Global-best checkpoint pair already exists")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("training_config") != config
            or metadata.get("actual_accuracy") != result["best_accuracy"]
            or metadata.get("trajectory") != result["trajectory"]
        ):
            raise RuntimeError("Existing global-best checkpoint metadata mismatch")
        return path
    state_dict = torch.load(result["best_checkpoint"], map_location="cpu")
    atomic_torch_save(state_dict, path)
    atomic_json_save(
        target_metadata(
            config,
            result["trajectory"],
            result["best_epoch"],
            result["best_accuracy"],
            path,
        ),
        metadata_path,
    )
    return path


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    validate_args(args)
    args.source = args.source.resolve()
    args.teacher = args.teacher.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {args.source}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output directory exists: {args.output_dir}; refusing to overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["QCFS_NUM_WORKERS"] = str(args.workers)
    configure_numerics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher, teacher_metadata = load_teacher_checkpoint(args.teacher, device)
    config = experiment_config(
        args, checkpoint_sha256(args.source), teacher_metadata
    )
    config_path = args.output_dir / "experiment_config.json"
    if config_path.is_file():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config:
            raise RuntimeError("Existing experiment configuration mismatch")
    else:
        atomic_json_save(config, config_path)

    seed_all(args.seed)
    _, test_loader = datapool(
        "cifar100", args.batch_size, augmentation_profile="paper_era"
    )
    source, source_metadata = load_student(args.source, device)
    source_accuracy = val(source, test_loader, device, 0)
    if abs(source_accuracy - args.expected_source_accuracy) > args.source_accuracy_tolerance:
        raise RuntimeError(
            f"Source accuracy gate failed: expected {args.expected_source_accuracy:.3f}% "
            f"+/- {args.source_accuracy_tolerance:.3f}, got {source_accuracy:.3f}%"
        )
    _, teacher_test_loader = datapool(
        "cifar100", 256, augmentation_profile="paper_era"
    )
    teacher_accuracy = val(teacher, teacher_test_loader, device, 0)
    if abs(teacher_accuracy - args.expected_teacher_accuracy) > args.teacher_accuracy_tolerance:
        raise RuntimeError(
            f"Teacher accuracy gate failed: expected {args.expected_teacher_accuracy:.3f}% "
            f"+/- {args.teacher_accuracy_tolerance:.3f}, got {teacher_accuracy:.3f}%"
        )
    print(
        f"Source accepted: {source_accuracy:.3f}% SHA256={source_metadata['sha256']}",
        flush=True,
    )
    print(
        f"Teacher accepted: {teacher_accuracy:.3f}% SHA256={teacher_metadata['sha256']}",
        flush=True,
    )
    del source
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = []
    for name, weight_lr, threshold_lr in zip(
        args.trajectory_names,
        args.weight_learning_rates,
        args.threshold_learning_rates,
    ):
        results.append(
            run_trajectory(
                args,
                config,
                name,
                weight_lr,
                threshold_lr,
                teacher,
                device,
            )
        )

    winner = max(results, key=lambda item: item["best_accuracy"])
    global_path = copy_global_best(
        winner, args.output_dir, config, args.resume
    )
    verified_global_accuracy, _ = verify_checkpoint(
        global_path, winner["best_accuracy"], test_loader, device
    )
    target_path = args.output_dir / TARGET_FILENAME
    verified_target_accuracy = None
    if target_path.is_file():
        target_info = json.loads(
            target_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        verified_target_accuracy, _ = verify_checkpoint(
            target_path, target_info["actual_accuracy"], test_loader, device
        )

    t32_accuracy = None
    t32_gap = None
    if verified_global_accuracy + 1e-12 >= args.target_accuracy:
        _, snn, _ = load_qcfs_pair(
            global_path, "cifar100", "resnet20", device
        )
        t32_accuracy, t32_gap = validate_t32_conversion(
            snn,
            test_loader,
            device,
            verified_global_accuracy,
            FORMAL_MAX_T32_CONVERSION_GAP,
        )

    summary = {
        "completed_at_utc": utc_now(),
        "experiment_config": config,
        "global_best": {
            "accuracy": verified_global_accuracy,
            "checkpoint": str(global_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256(global_path),
            "trajectory": winner["trajectory"],
        },
        "source_accuracy": source_accuracy,
        "target_checkpoint": str(target_path.resolve()) if target_path.is_file() else None,
        "target_reached": verified_target_accuracy is not None,
        "teacher_accuracy": teacher_accuracy,
        "t32_accuracy": t32_accuracy,
        "t32_conversion_gap": t32_gap,
        "t32_gate_passed": t32_gap is not None and t32_gap <= FORMAL_MAX_T32_CONVERSION_GAP,
        "trajectories": results,
    }
    atomic_json_save(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
