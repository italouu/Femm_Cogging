import glob
import torch
from src.configs.training import NnCfg
from src.neural_op.archs import ARCH_REGISTRY
from src.neural_op.dataloaders.loaders import build_loaders, CUDAPrefetcher
from src.neural_op.losses import LOSS_REGISTRY
from src.neural_op.normalization import Normalizer
from src.neural_op.training_utils import fit
from src.neural_op.monitor import TrainingMonitor
from src.training.model_manager import ModelManager

# ── Configuração ───────────────────────────────────────────────────────────────

def run(_nn):
    """
    Executa um treino completo a partir de um NnCfg já construído. Chamado pelo
    `__main__` abaixo e por scripts/run_battery.py (sequência de treinos), sem
    duplicar o pipeline model/optimizer/scheduler/fit/ModelManager.
    """
    entry       = ARCH_REGISTRY[_nn.arch]
    chunk_paths = sorted(glob.glob(
        f'data/torch/data_chunks/{_nn.dataset}/data_chunk_*.pt'
    ))

    train_loader, test_loader, train_paths, test_paths = build_loaders(
        chunk_paths,
        batch_size      = _nn.batch_size,
        train_split     = _nn.train_split,
        buffer_size     = _nn.buffer_size,
        num_workers     = _nn.num_workers,
        prefetch_factor = _nn.prefetch_factor,
        seed            = _nn.split_seed,
        mode            = entry.loader_mode,
        test_split      = _nn.test_split,
    )

    # normalizer : None se _nn.normalize=False; senão reconstruído das stats já
    # calculadas em NnCfg.__post_init__ (Normalizer.fit, com cache em disco).
    # Atrelado ao modelo (model.normalizer) logo após make_model — a metric_fn
    # lê de lá pra decodificar pred/y de volta pra unidade física.
    normalizer = Normalizer.from_dict(_nn.norm_stats) if _nn.normalize and _nn.norm_stats else None

    step_fn   = entry.make_step_fn(_nn.arch_cfg, _nn.loss_cfg)
    metric_fn = entry.metric_fn(_nn.arch_cfg)
    # loss_obj (BaseLoss, src/neural_op/losses.py): loss_fn é o próprio objeto;
    # log_fn sai de loss_obj.log_epoch.
    loss_obj = LOSS_REGISTRY[_nn.loss]
    loss_fn  = loss_obj

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

    model.normalizer = normalizer

    # DEVICE dentro de run (não em nível de módulo): is_available() inicializa o CUDA, e após
    # um crash de OOM o driver pode travar aqui sem nenhum print -- só depois do log inicial.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}", flush=True)

    train_loader = CUDAPrefetcher(train_loader, DEVICE, normalizer=normalizer)
    test_loader  = CUDAPrefetcher(test_loader,  DEVICE, normalizer=normalizer)

    mgr     = ModelManager(_nn)
    mgr.open(model, DEVICE, resumed_from=_nn.resume_run,
             split={'train': train_paths, 'test': test_paths})
    monitor = TrainingMonitor(cfg=_nn.monitor_cfg, mgr=mgr)

    status = 'failed'
    stop_reason = 'exception'   # B4 — sobrescrito abaixo se o fit terminar/parar normalmente
    try:
        model, losses = fit(
            model, train_loader, test_loader,
            optimizer, scheduler, loss_fn,
            device=DEVICE, n_epochs=_nn.n_epochs,
            step_fn=step_fn,
            metric_fn=metric_fn,
            log_fn=loss_obj.log_epoch,
            monitor=monitor,
            start_epoch=start_epoch,
            prev_losses=prev_losses,
            epoch_log_fn=mgr.log_epoch,   # B4 — epochs.csv
        )
        status = 'stopped' if monitor.stopped_early else 'done'
        # B4 — 'gl' | 'early_stop_patience' (monitor) ou 'n_epochs' (rodou até o fim)
        stop_reason = monitor.stop_reason or 'n_epochs'
    except KeyboardInterrupt:
        status = 'stopped'
        stop_reason = 'keyboard_interrupt'
    finally:
        mgr.close(status, monitor.last_epoch, model, optimizer, scheduler,
                  stop_reason=stop_reason, best_epoch=monitor.best_epoch,
                  best_test_loss=monitor.best_loss, n_epochs_cfg=_nn.n_epochs)
    return status


if __name__ == '__main__':
    # Execução avulsa = defaults de src/configs (run elem_mae da bateria bldc).
    # NnCfg só aqui (não em nível de módulo): o __post_init__ detecta dims e roda
    # Normalizer.fit -- não pode ser efeito colateral de importar `run` (run_battery).
    _nn = NnCfg()
    run(_nn)

