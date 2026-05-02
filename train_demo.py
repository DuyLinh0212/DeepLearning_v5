import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from config import config as base_config
from models import Densenet121, EfficientNetB0, EfficientNetB0_SA
from utils import _get_lr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _build_model(name: str):
    name = name.lower()
    if name == "densenet121":
        return Densenet121()
    if name == "efficientnetb0":
        return EfficientNetB0()
    if name == "efficientnetb0_sa":
        return EfficientNetB0_SA()
    raise ValueError(f"Unsupported model: {name}")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    phase="train",
    scaler=None,
    use_amp=False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    y_true = []
    y_prob = []
    losses = []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=phase, leave=False)

    for batch in iterator:
        if batch is None:
            continue
        images, label = batch

        if device != "cpu":
            images = [img.to(device) for img in images]
            label = label.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=bool(use_amp and device != "cpu")):
                output = model(images)
                loss = criterion(output, label)
            if is_train:
                if scaler is not None and bool(use_amp and device != "cpu"):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(loss.item())

        probas = torch.sigmoid(output).detach().cpu().view(-1).numpy().tolist()
        labels = label.detach().cpu().view(-1).numpy().tolist()

        y_prob.extend(probas)
        y_true.extend(labels)

    if len(losses) == 0:
        return 0.0, [], []

    loss_mean = float(np.mean(losses))
    return loss_mean, y_true, y_prob


def _compute_metrics(y_true, y_prob, threshold=0.5):
    if len(y_true) == 0:
        return {
            "auc": 0.5,
            "acc": 0.0,
            "se": 0.0,
            "sp": 0.0,
            "balanced_acc": 0.0,
            "threshold": float(threshold),
            "y_pred": [],
        }

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    se = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    sp = 0.0 if (tn + fp) == 0 else tn / (tn + fp)

    return {
        "auc": float(auc),
        "acc": float(metrics.accuracy_score(y_true, y_pred)),
        "se": float(se),
        "sp": float(sp),
        "balanced_acc": float((se + sp) / 2.0),
        "threshold": float(threshold),
        "y_pred": y_pred,
    }


def _find_best_threshold_by_balanced_acc(y_true, y_prob, num_thresholds=101):
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return 0.5, 0.0

    best_threshold = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.0, 1.0, num=num_thresholds)
    for threshold in thresholds:
        m = _compute_metrics(y_true, y_prob, threshold=float(threshold))
        score = m["balanced_acc"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def _append_csv(csv_path, row, header):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _ensure_csv_header(csv_path, header):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return
    existing_header = first_line.split(",")
    if existing_header == header:
        return
    backup_path = f"{csv_path}.bak_{int(time.time())}"
    os.replace(csv_path, backup_path)
    print(f"Detected old metrics CSV format. Backed up to: {backup_path}")


def _plot_curves(csv_path, out_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    epochs = np.atleast_1d(data["epoch"])
    train_loss = np.atleast_1d(data["train_loss"])
    val_loss = np.atleast_1d(data["val_loss"])
    train_auc = np.atleast_1d(data["train_auc"])
    val_auc = np.atleast_1d(data["val_auc"])
    train_acc = np.atleast_1d(data["train_acc"])
    val_acc = np.atleast_1d(data["val_acc"])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, train_auc, label="train_auc")
    plt.plot(epochs, val_auc, label="val_auc")
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, out_path):
    if len(y_true) == 0:
        return
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_roc(y_true, y_prob, out_path):
    if len(y_true) == 0:
        return
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        auc = metrics.auc(fpr, tpr)
    except Exception:
        return
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train(
    config: dict,
    model_name: str,
    data_root: str = "data",
    labels_root: str = "labels",
    run_idx: int = 1,
    run_seed: int = 42,
):
    _set_seed(run_seed)
    run_name = f"run_{run_idx:02d}"

    save_folder = os.path.join("weights", config["task"], run_name)
    os.makedirs(save_folder, exist_ok=True)

    eval_folder = os.path.join("evaluation", f"{model_name}_{config['task']}", run_name)
    os.makedirs(eval_folder, exist_ok=True)

    csv_path = os.path.join(eval_folder, f"{model_name}_{config['task']}_metrics.csv")
    best_model_path = os.path.join(save_folder, f"{model_name}_best_model.pth")
    last_model_path = os.path.join(save_folder, f"{model_name}_last_checkpoint.pth")

    print(f"Starting to Train Model... task={config['task']} | {run_name} | seed={run_seed}")
    train_loader, val_loader, test_loader, train_wts, val_wts, test_wts = load_data(
        config["task"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        target_slices=config["target_slices"],
        image_size=config["image_size"],
        data_root=data_root,
        label_root=labels_root,
        include_test=True,
    )

    print("Initializing Model...")
    model = _build_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
        train_wts = train_wts.cuda()
        val_wts = val_wts.cuda()
        if test_wts is not None:
            test_wts = test_wts.cuda()

    print("Initializing Loss Method...")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)
    test_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=test_wts) if test_wts is not None else val_criterion
    if device == "cuda":
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()
        if test_wts is not None:
            test_criterion = test_criterion.cuda()

    print("Setup the Optimizer")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config.get("momentum", 0.7),
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.3, threshold=1e-4
    )
    use_amp = bool(device == "cuda")
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP enabled: {use_amp}")

    starting_epoch = config["starting_epoch"]
    num_epochs = config["max_epoch"]
    best_val_auc = float(0)
    patience = config.get("patience", 5)
    epochs_no_improve = 0

    print("Run starts from scratch (no abnormal warm-start, no checkpoint resume).")

    writer = SummaryWriter(comment=f"model={model_name} lr={config['lr']} task={config['task']} {run_name}")
    t_start_training = time.time()

    header = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_acc",
        "train_se",
        "train_sp",
        "train_balanced_acc",
        "val_loss",
        "val_auc",
        "val_acc",
        "val_se",
        "val_sp",
        "val_balanced_acc",
        "val_best_threshold",
        "val_best_balanced_acc",
        "val_best_se",
        "val_best_sp",
        "val_best_acc",
        "lr",
    ]
    _ensure_csv_header(csv_path, header)

    for epoch in range(starting_epoch, num_epochs):
        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()

        train_loss, train_true, train_prob = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            phase="train",
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_true, val_prob = _run_epoch(
            model,
            val_loader,
            val_criterion,
            optimizer=None,
            device=device,
            phase="val",
            scaler=scaler,
            use_amp=use_amp,
        )

        train_metrics = _compute_metrics(train_true, train_prob, threshold=0.5)
        val_metrics = _compute_metrics(val_true, val_prob, threshold=0.5)
        val_best_threshold, _ = _find_best_threshold_by_balanced_acc(val_true, val_prob)
        val_best_metrics = _compute_metrics(val_true, val_prob, threshold=val_best_threshold)

        writer.add_scalar("Train/Avg Loss", train_loss, epoch)
        writer.add_scalar("Train/AUC_epoch", train_metrics["auc"], epoch)
        writer.add_scalar("Train/Acc_epoch", train_metrics["acc"], epoch)
        writer.add_scalar("Train/SE_epoch", train_metrics["se"], epoch)
        writer.add_scalar("Train/SP_epoch", train_metrics["sp"], epoch)
        writer.add_scalar("Train/BalancedAcc_epoch", train_metrics["balanced_acc"], epoch)
        writer.add_scalar("Val/Avg Loss", val_loss, epoch)
        writer.add_scalar("Val/AUC_epoch", val_metrics["auc"], epoch)
        writer.add_scalar("Val/Acc_epoch", val_metrics["acc"], epoch)
        writer.add_scalar("Val/SE_epoch", val_metrics["se"], epoch)
        writer.add_scalar("Val/SP_epoch", val_metrics["sp"], epoch)
        writer.add_scalar("Val/BalancedAcc_epoch", val_metrics["balanced_acc"], epoch)
        writer.add_scalar("Val/BestThreshold_BalancedAcc", val_best_threshold, epoch)
        writer.add_scalar("Val/BestBalancedAcc_epoch", val_best_metrics["balanced_acc"], epoch)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time
        print(
            "Epoch [{}/{}] | train loss {:.4f} | train auc {:.4f} | train acc {:.4f} | "
            "train se/sp {:.4f}/{:.4f} | val loss {:.4f} | val auc {:.4f} | "
            "val se/sp@0.5 {:.4f}/{:.4f} | val best_thr {:.2f} bal_acc {:.4f} | time {:.2f} s".format(
                epoch,
                num_epochs,
                train_loss,
                train_metrics["auc"],
                train_metrics["acc"],
                train_metrics["se"],
                train_metrics["sp"],
                val_loss,
                val_metrics["auc"],
                val_metrics["se"],
                val_metrics["sp"],
                val_best_threshold,
                val_best_metrics["balanced_acc"],
                delta,
            )
        )
        print("-" * 30)
        writer.flush()

        _append_csv(
            csv_path,
            [
                epoch,
                train_loss,
                train_metrics["auc"],
                train_metrics["acc"],
                train_metrics["se"],
                train_metrics["sp"],
                train_metrics["balanced_acc"],
                val_loss,
                val_metrics["auc"],
                val_metrics["acc"],
                val_metrics["se"],
                val_metrics["sp"],
                val_metrics["balanced_acc"],
                val_best_threshold,
                val_best_metrics["balanced_acc"],
                val_best_metrics["se"],
                val_best_metrics["sp"],
                val_best_metrics["acc"],
                current_lr,
            ],
            header,
        )

        improved = val_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = val_metrics["auc"]
            epochs_no_improve = 0
            print(f"*** New Best AUC: {best_val_auc:.4f}. Saving best model for {model_name}...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "model_name": model_name,
                },
                best_model_path,
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "model_name": model_name,
            },
            last_model_path,
        )
        print(f"Checkpoint saved to {last_model_path}")

        if not improved:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement in {patience} epochs.")
            break

    t_end_training = time.time()
    print(f"Training finished. Total time: {t_end_training - t_start_training:.2f} s")
    writer.flush()
    writer.close()

    # Load best model for final evaluation/plots
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    _, val_true, val_prob = _run_epoch(
        model,
        val_loader,
        val_criterion,
        optimizer=None,
        device=device,
        phase="val",
        scaler=scaler,
        use_amp=use_amp,
    )

    best_threshold, _ = _find_best_threshold_by_balanced_acc(val_true, val_prob)
    val_final_metrics = _compute_metrics(val_true, val_prob, threshold=best_threshold)
    print(
        "Final VALID metrics | thr {:.2f} | auc {:.4f} | acc {:.4f} | se {:.4f} | sp {:.4f} | bal_acc {:.4f}".format(
            best_threshold,
            val_final_metrics["auc"],
            val_final_metrics["acc"],
            val_final_metrics["se"],
            val_final_metrics["sp"],
            val_final_metrics["balanced_acc"],
        )
    )

    _plot_curves(csv_path, os.path.join(eval_folder, f"{model_name}_{config['task']}_curves.png"))
    _plot_confusion_matrix(
        val_true,
        val_final_metrics["y_pred"],
        os.path.join(eval_folder, f"{model_name}_{config['task']}_confusion.png"),
    )
    _plot_roc(val_true, val_prob, os.path.join(eval_folder, f"{model_name}_{config['task']}_roc.png"))

    test_metrics_csv = os.path.join(eval_folder, f"{model_name}_{config['task']}_test_metrics.csv")
    if test_loader is not None:
        _, test_true, test_prob = _run_epoch(
            model,
            test_loader,
            test_criterion,
            optimizer=None,
            device=device,
            phase="test",
            scaler=scaler,
            use_amp=use_amp,
        )
        test_metrics = _compute_metrics(test_true, test_prob, threshold=best_threshold)
        print(
            "Final TEST metrics  | thr {:.2f} | auc {:.4f} | acc {:.4f} | se {:.4f} | sp {:.4f} | bal_acc {:.4f}".format(
                best_threshold,
                test_metrics["auc"],
                test_metrics["acc"],
                test_metrics["se"],
                test_metrics["sp"],
                test_metrics["balanced_acc"],
            )
        )
        _append_csv(
            test_metrics_csv,
            [
                config["task"],
                run_name,
                best_threshold,
                test_metrics["auc"],
                test_metrics["acc"],
                test_metrics["se"],
                test_metrics["sp"],
                test_metrics["balanced_acc"],
            ],
            ["task", "run", "threshold", "auc", "acc", "se", "sp", "balanced_acc"],
        )
        _plot_confusion_matrix(
            test_true,
            test_metrics["y_pred"],
            os.path.join(eval_folder, f"{model_name}_{config['task']}_test_confusion.png"),
        )
        _plot_roc(test_true, test_prob, os.path.join(eval_folder, f"{model_name}_{config['task']}_test_roc.png"))
    else:
        print("Skip TEST evaluation: test split not found.")

    print(f"Metrics saved to: {csv_path}")
    if test_loader is not None:
        print(f"Test metrics saved to: {test_metrics_csv}")
    print(f"Plots saved to: {eval_folder}")
    return {
        "task": config["task"],
        "run": run_name,
        "threshold": best_threshold,
        "val_auc": val_final_metrics["auc"],
        "val_acc": val_final_metrics["acc"],
        "val_se": val_final_metrics["se"],
        "val_sp": val_final_metrics["sp"],
        "val_balanced_acc": val_final_metrics["balanced_acc"],
        "test_auc": (test_metrics["auc"] if test_loader is not None else np.nan),
        "test_acc": (test_metrics["acc"] if test_loader is not None else np.nan),
        "test_se": (test_metrics["se"] if test_loader is not None else np.nan),
        "test_sp": (test_metrics["sp"] if test_loader is not None else np.nan),
        "test_balanced_acc": (test_metrics["balanced_acc"] if test_loader is not None else np.nan),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnetb0",
        choices=["densenet121", "efficientnetb0", "efficientnetb0_sa"],
        help="Choose model to train",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="abnormal,acl,meniscus",
        help="Comma-separated tasks to train (default: abnormal,acl,meniscus)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Directory containing train/valid/test MRI folders (default: ./data).",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default="labels",
        help="Directory containing train-*.csv, valid-*.csv and test-*.csv (default: ./labels).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=int(base_config.get("num_runs", 10)),
        help="Number of independent runs per task (default: 10).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=int(base_config.get("base_seed", 42)),
        help="Base random seed. Each run uses base_seed + run_idx.",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        cfg = dict(base_config)
        cfg["task"] = task
        print("Training Configuration")
        print(cfg)
        run_results = []
        for run_idx in range(1, args.num_runs + 1):
            run_seed = args.base_seed + run_idx
            result = train(
                config=cfg,
                model_name=args.model,
                data_root=args.data_root,
                labels_root=args.labels_root,
                run_idx=run_idx,
                run_seed=run_seed,
            )
            run_results.append(result)

        summary_path = os.path.join("evaluation", f"{args.model}_{task}", f"{args.model}_{task}_runs_summary.csv")
        summary_header = [
            "task",
            "run",
            "threshold",
            "val_auc",
            "val_acc",
            "val_se",
            "val_sp",
            "val_balanced_acc",
            "test_auc",
            "test_acc",
            "test_se",
            "test_sp",
            "test_balanced_acc",
        ]
        if os.path.exists(summary_path):
            os.remove(summary_path)
        for row in run_results:
            _append_csv(
                summary_path,
                [
                    row["task"],
                    row["run"],
                    row["threshold"],
                    row["val_auc"],
                    row["val_acc"],
                    row["val_se"],
                    row["val_sp"],
                    row["val_balanced_acc"],
                    row["test_auc"],
                    row["test_acc"],
                    row["test_se"],
                    row["test_sp"],
                    row["test_balanced_acc"],
                ],
                summary_header,
            )

        test_auc_vals = [r["test_auc"] for r in run_results if not np.isnan(r["test_auc"])]
        val_auc_vals = [r["val_auc"] for r in run_results]
        if test_auc_vals:
            print(
                "Task {} | {} runs | mean+-std val_auc {:.4f}+-{:.4f} | test_auc {:.4f}+-{:.4f}".format(
                    task,
                    args.num_runs,
                    float(np.mean(val_auc_vals)),
                    float(np.std(val_auc_vals)),
                    float(np.mean(test_auc_vals)),
                    float(np.std(test_auc_vals)),
                )
            )
        else:
            print(
                "Task {} | {} runs | mean+-std val_auc {:.4f}+-{:.4f} | test split not found".format(
                    task,
                    args.num_runs,
                    float(np.mean(val_auc_vals)),
                    float(np.std(val_auc_vals)),
                )
            )
        print(f"Runs summary saved to: {summary_path}")
    print("Training Ended...")
