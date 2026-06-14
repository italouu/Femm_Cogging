import glob
import torch
from src.configs.training import NnCfg
from src.neural_op.archs import ARCH_REGISTRY
from src.neural_op.dataloaders.grid_loader import build_loaders, CUDAPrefetcher
from src.neural_op.losses import LOSS_REGISTRY
from src.neural_op.training_utils import fit
from src.neural_op.monitor import TrainingMonitor
from src.training.model_manager import ModelManager

# ── Configuração ───────────────────────────────────────────────────────────────
from src.configs.training import FNOConfig, FNORefConfig
from src.configs.monitor import MonitorCfg

_nn = NnCfg()

# [REMOVIDO] DEVICE em nível de módulo — torch.cuda.is_available() inicializa o
# contexto CUDA imediatamente, antes de qualquer print. Após crash de OOM, o
# driver NVIDIA pode ficar em estado inconsistente e travar aqui sem nenhum
# output visível no console. Movido para dentro do if __name__ abaixo.
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    entry       = ARCH_REGISTRY[_nn.arch]
    chunk_paths = sorted(glob.glob(
        f'data/torch/data_chunks/{_nn.dataset}/data_chunk_*.pt'
    ))

    train_loader, test_loader = build_loaders(
        chunk_paths,
        batch_size      = _nn.batch_size,
        train_split     = _nn.train_split,
        buffer_size     = _nn.buffer_size,
        num_workers     = _nn.num_workers,
        prefetch_factor = _nn.prefetch_factor,
        seed            = _nn.split_seed,
        mode            = entry.loader_mode,
    )

    step_fn = entry.make_step_fn(_nn.arch_cfg)
    loss_fn = LOSS_REGISTRY[_nn.loss]

    # ── Modelo, optimizer, scheduler ───────────────────────────────────────
    if _nn.resume_run:
        # ── resume ─────────────────────────────────────────────────────────
        ckpt, prev_losses = ModelManager.load_run(_nn.resume_run, _nn.resume_checkpoint)
        start_epoch = ckpt['epoch'] + 1
        model     = entry.make_model(_nn.arch_cfg)
        sd        = {k: v for k, v in ckpt['model_state_dict'].items() if k != '_metadata'}
        model.load_state_dict(sd)
        optimizer = torch.optim.Adam(model.parameters(), lr=_nn.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=_nn.scheduler_step, gamma=_nn.scheduler_gamma)
        if not _nn.resume_modified:
            # resume limpo: estado completo do optimizer e scheduler
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            # resume com modificação (ex: nova loss, novo lr): só pesos carregados
            print("  resume modificado: optimizer e scheduler reiniciados do zero")
    else:
        # ── treino do zero ──────────────────────────────────────────────────
        start_epoch = 0
        prev_losses = None
        model     = entry.make_model(_nn.arch_cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=_nn.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=_nn.scheduler_step, gamma=_nn.scheduler_gamma)

    # [REMOVIDO] ckpt_extra migrado para config.json via ModelManager
    # ckpt_extra = {
    #     'config':          NET_CONFIG,
    #     'optimizer_label': type(optimizer).__name__,
    #     'loss_label':      _nn.loss,
    # }

    # Inicializa CUDA aqui (e não no nível de módulo) para que os prints de
    # build_loaders apareçam antes — diagnóstico mais claro se o driver CUDA
    # estiver em estado inválido após crash de OOM.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}", flush=True)

    train_loader = CUDAPrefetcher(train_loader, DEVICE)
    test_loader  = CUDAPrefetcher(test_loader,  DEVICE)

    mgr     = ModelManager(_nn)
    mgr.open(model, DEVICE, resumed_from=_nn.resume_run)
    monitor = TrainingMonitor(cfg=_nn.monitor_cfg, mgr=mgr)

    status = 'failed'
    try:
        model, losses = fit(
            model, train_loader, test_loader,
            optimizer, scheduler, loss_fn,
            device=DEVICE, n_epochs=_nn.n_epochs,
            step_fn=step_fn,
            monitor=monitor,
            start_epoch=start_epoch,
            prev_losses=prev_losses,
        )
        status = 'stopped' if monitor.stopped_early else 'done'
    except KeyboardInterrupt:
        status = 'stopped'
    finally:
        mgr.close(status, monitor.last_epoch, model, optimizer, scheduler)

    # [REMOVIDO] save interativo substituído por mgr.close() que salva model_final.pth
    # name = input('save? (nome ou Enter para pular): ').strip()
    # if name:
    #     path = f'data/models/{_nn.dataset}/{name}.pth'
    #     pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    #     save_checkpoint(path, start_epoch + _nn.n_epochs - 1,
    #                     model, optimizer, scheduler,
    #                     losses['train'], losses['test'], ckpt_extra)
    #     print(f"Salvo em {path}")
