"""
Stage 2 only: FixRes fine-tune of DoGDepthNet.

Loads the best Stage-1 checkpoint (cfg.SAVE_PATH) and runs the 480x640
fine-tune for cfg.NUM_FT_EPOCHS epochs, saving the best to cfg.SAVE_PATH_FT.

Full-state resume is automatic via cfg.LATEST_PATH_FT: kill and restart
this script any time and it will continue from the last completed epoch
with scheduler / scaler / EMA-step counters restored.
"""
import os

import torch

from losses.dog_losses import DoGDepthLoss
from train_nyu import (
    Config,
    build_model,
    make_ema,
    run_stage,
)

cfg = Config()


def main():
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    assert os.path.exists(cfg.SAVE_PATH), (
        f'Stage-1 best checkpoint not found at {cfg.SAVE_PATH}. '
        f'Run `python train_nyu.py` first (Stage 1) and try again.'
    )

    model = build_model(device)
    ema = make_ema(model)
    criterion = DoGDepthLoss(
        alpha_si=1.0,
        alpha_berhu=0.05,
        alpha_log10=0.025,
        beta_grad=0.25,
        beta_vnl=0.1,
        gamma=0.05,
        delta=0.05,
    )

    if not os.path.exists(cfg.LATEST_PATH_FT):
        ck = torch.load(cfg.SAVE_PATH, map_location=device)
        model.load_state_dict(ck['model_state'])
        ema.load_state_dict(ck['ema_state'])
        stage1_a1 = ck.get('val_metrics', {}).get('a1', float('nan'))
        stage1_epoch = ck.get('epoch', '?')
        print(f'[init] Loaded Stage-1 best: epoch={stage1_epoch} '
              f'a1={stage1_a1:.4f} from {cfg.SAVE_PATH}')
    else:
        print(f'[init] Stage-2 latest checkpoint found at {cfg.LATEST_PATH_FT}; '
              f'run_stage() will resume from it.')

    run_stage(
        stage_name='fixres-480x640',
        model=model,
        ema=ema,
        criterion=criterion,
        optim_kwargs=dict(max_lr=cfg.MAX_LR / 10.0),
        loader_kwargs=dict(
            mode='h5',
            h5_train_dir=cfg.H5_TRAIN_DIR,
            h5_val_dir=cfg.H5_VAL_DIR,
            img_size=cfg.FT_IMG_SIZE,
            batch_size=max(2, cfg.BATCH_SIZE // 2),
            num_workers=cfg.NUM_WORKERS,
        ),
        save_path=cfg.SAVE_PATH_FT,
        num_epochs=cfg.NUM_FT_EPOCHS,
        device=device,
        resume_path=cfg.SAVE_PATH,
        latest_path=cfg.LATEST_PATH_FT,
    )


if __name__ == '__main__':
    main()
