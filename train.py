import os
import time
import torch
import torch.utils.data as data
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from models import MRnet
from config import config
from utils import _train_model, _evaluate_model, _get_lr

"""Performs training of a specified model.
    
Input params:
    config_file: Takes in configurations to train with 
"""


def _parse_gpu_ids(gpu_ids_cfg, num_available):
    if num_available <= 0:
        return []

    if gpu_ids_cfg is None:
        return list(range(num_available))

    if isinstance(gpu_ids_cfg, str):
        gpu_ids_cfg = [x.strip() for x in gpu_ids_cfg.split(",") if x.strip()]
    if isinstance(gpu_ids_cfg, int):
        gpu_ids_cfg = [gpu_ids_cfg]

    parsed = []
    for gpu_id in gpu_ids_cfg:
        try:
            idx = int(gpu_id)
        except Exception:
            continue
        if 0 <= idx < num_available:
            parsed.append(idx)
    if not parsed:
        return list(range(num_available))
    return sorted(list(dict.fromkeys(parsed)))


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _load_flexible_state_dict(model, state_dict):
    target_model = _unwrap_model(model)
    target_keys = target_model.state_dict().keys()
    target_has_module = next(iter(target_keys)).startswith('module.')
    src_has_module = next(iter(state_dict.keys())).startswith('module.')

    if src_has_module and not target_has_module:
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    elif not src_has_module and target_has_module:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    target_model.load_state_dict(state_dict)

def train(config : dict):
    """
    Function where actual training takes place

    Args:
        config (dict) : Configuration to train with
    """
    
    # 1. SETUP THƯ MỤC LƯU WEIGHTS
    save_folder = './weights/{}'.format(config['task'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Directory {save_folder} created!")

    # Định nghĩa đường dẫn file
    checkpoint_path = os.path.join(save_folder, 'last_checkpoint.pth')
    best_model_path = os.path.join(save_folder, 'best_model.pth')

    print('Starting to Train Model...')
    train_loader, val_loader, train_wts, val_wts = load_data(
        config['task'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_slices=config['target_slices'],
        image_size=config['image_size'],
    )

    print('Initializing Model...')
    model = MRnet()
    device = 'cpu'
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = bool(config.get('use_data_parallel', True))
    gpu_ids = _parse_gpu_ids(config.get('gpu_ids', None), num_gpus)

    if num_gpus > 0:
        primary_gpu = gpu_ids[0] if gpu_ids else 0
        device = f'cuda:{primary_gpu}'
        model = model.to(device)
        if use_data_parallel and len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=primary_gpu)
            print(f'DataParallel enabled on GPUs: {gpu_ids}')
        else:
            print(f'DataParallel disabled. Using single GPU (available={num_gpus}, selected={gpu_ids[:1]}).')
        train_wts = train_wts.to(device)
        val_wts = val_wts.to(device)
    else:
        print('Running on CPU.')

    print('Initializing Loss Method...')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_wts)

    if device != 'cpu':
        criterion = criterion.to(device)
        val_criterion = val_criterion.to(device)

    print('Setup the Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=.3, threshold=1e-4
    )
    use_amp = bool(config.get('use_amp', True) and device != 'cpu')
    scaler = GradScaler(enabled=use_amp)
    print(f'AMP enabled: {use_amp}')
    
    # Các tham số mặc định
    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']
    best_val_auc = float(0)
    
    # --- ĐOẠN CODE MỚI: KIỂM TRA ĐỂ RESUME (CHẠY TIẾP) ---
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        _load_flexible_state_dict(model, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1 # Bắt đầu từ epoch tiếp theo
        best_val_auc = checkpoint['best_val_auc']
        
        print(f"Resuming training from epoch {starting_epoch} with Best AUC: {best_val_auc}")
    else:
        print("No checkpoint found. Starting from scratch.")
    # -----------------------------------------------------

    patience = config['patience']
    log_train = config['log_train']
    log_val = config['log_val']

    print('Starting Training')

    writer = SummaryWriter(comment='lr={} task={}'.format(config['lr'], config['task']))
    t_start_training = time.time()

    # Vòng lặp chính
    for epoch in range(starting_epoch, num_epochs):

        current_lr = _get_lr(optimizer)
        epoch_start_time = time.time()

        # Train
        train_loss, train_auc = _train_model(
            model,
            train_loader,
            epoch,
            num_epochs,
            optimizer,
            criterion,
            writer,
            current_lr,
            log_train,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )

        # Evaluate
        val_loss, val_auc = _evaluate_model(
            model,
            val_loader,
            val_criterion,
            epoch,
            num_epochs,
            writer,
            current_lr,
            log_val,
            device=device,
            use_amp=use_amp,
        )

        # Log tensorboard
        writer.add_scalar('Train/Avg Loss', train_loss, epoch)
        writer.add_scalar('Val/Avg Loss', val_loss, epoch)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time

        print("Epoch [{}/{}] | train loss : {:.4f} | train auc {:.4f} | val loss {:.4f} | val auc {:.4f} | time {:.2f} s".format(
            epoch, num_epochs, train_loss, train_auc, val_loss, val_auc, delta))

        print('-' * 30)
        writer.flush()

        # --- LƯU MODEL TỐT NHẤT (BEST) ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"*** New Best AUC: {best_val_auc:.4f}. Saving best model...")
            torch.save({
                'model_state_dict': _unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_val_auc,
            }, best_model_path)
        
        # --- LƯU CHECKPOINT ĐỂ RESUME SAU NÀY (LAST) ---
        # Lưu đè lên file này mỗi cuối epoch
        torch.save({
            'model_state_dict': _unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_auc': best_val_auc, # Lưu lại kỷ lục hiện tại
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        # -----------------------------------------------

    t_end_training = time.time()
    print(f'Training finished. Total time: {t_end_training - t_start_training:.2f} s')
    writer.flush()
    writer.close()

if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
