import torch
import time


def count_params_real(model, trainable_only=False):
    """Nº de parâmetros REAIS: tensores complexos (pesos espectrais cfloat do
    FNO) contam em dobro (parte real + imaginária)."""
    c = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        c += p.numel() * (2 if p.is_complex() else 1)
    return c


def _batch_size(batch):
    """Conta amostras no batch independente do formato (tuple grid ou dict de grafo, loaders 'vertex'/'bipartite')."""
    if isinstance(batch, (tuple, list)):
        return int(batch[0].shape[0])
    return int(batch['x_hw'].shape[0])


def train_epoch(model, loader, optimizer, loss_fn, device, step_fn):
    """
    Executa uma época de treino.

    Parameters
    ----------
    step_fn : callable
        step_fn(batch, model, loss_fn, device) -> loss (scalar tensor)
        Definido no script de treino; encapsula o forward específico do modelo.

    Returns
    -------
    avg_loss  : float
    n_samples : int  — total de amostras processadas (para samples_per_s)
    """
    model.train()
    # Acumula no GPU — evita .item() por batch (sync CPU↔GPU a cada iteração)
    total_loss = torch.zeros(1, device=device)
    n_batches  = 0
    n_samples  = 0
    for batch in loader:
        n_samples += _batch_size(batch)
        optimizer.zero_grad()
        loss = step_fn(batch, model, loss_fn, device)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.detach()
        n_batches  += 1
    # Contador em vez de len(loader) — IterableDataset não implementa __len__
    # .item() único por época — único sync point com o CPU
    return (total_loss.item() / n_batches) if n_batches > 0 else 0.0, n_samples


def eval_epoch(model, loader, loss_fn, device, step_fn):
    """
    Executa uma época de avaliação sem gradiente.

    Parameters
    ----------
    step_fn : callable
        Mesmo contrato de train_epoch.
    """
    model.eval()
    total_loss = torch.zeros(1, device=device)
    n_batches  = 0
    with torch.no_grad():
        for batch in loader:
            loss = step_fn(batch, model, loss_fn, device)
            total_loss += loss.detach()
            n_batches  += 1
    return (total_loss.item() / n_batches) if n_batches > 0 else 0.0


def compute_mae_metrics(model, loader, device, metric_fn):
    """
    Uma passagem sobre loader sem gradiente, acumulando mae_hw/mae_graph (MAE bruto,
    sem máscara) via metric_fn. Chamada só no heartbeat (dentro de fit()) — custo
    extra de uma passada a mais no ritmo de checkpoint_every, não em toda época.

    metric_fn : (batch, model, device) -> (mae_hw: float, mae_graph: float|None)

    Returns
    -------
    mae_hw_avg    : float — média por batch de mae_hw
    mae_graph_avg : float|None — média por batch de mae_graph; None se o arch não
                    produz saída em grafo (todo mae_graph do loader é None)
    """
    model.eval()
    total_hw, total_graph = 0.0, 0.0
    has_graph = False
    has_hw = False      # archs com GNN devolvem mae_hw=None
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            mae_hw, mae_graph = metric_fn(batch, model, device)
            if mae_hw is not None:
                total_hw += mae_hw
                has_hw = True
            if mae_graph is not None:
                total_graph += mae_graph
                has_graph = True
            n_batches += 1
    if n_batches == 0:
        return 0.0, None
    mae_graph_avg = (total_graph / n_batches) if has_graph else None
    return (total_hw / n_batches) if has_hw else None, mae_graph_avg


def _default_log_epoch(ep, train_loss, val_loss, train_time_s, eval_time_s, samples_per_s):
    """log_epoch padrão -- usado se fit() for chamado sem log_fn ou por qualquer
    BaseLoss que não sobrescreva log_epoch (src/neural_op/losses.py)."""
    print(f"epoch {ep:>4d}  train {train_loss:.4e}  val {val_loss:.4e}"
          f"  [{train_time_s:.1f}s + {eval_time_s:.1f}s eval]  {samples_per_s:.0f} samp/s")


def save_checkpoint(path, epoch, model, optimizer, scheduler
                    ):
    """Salva estado de treino: pesos, otimizador, scheduler e época."""
    # strip de '_metadata': neuralop.FNO inclui essa chave no state_dict para
    # serializar hiperparâmetros; load_state_dict a rejeita como chave inesperada
    sd = {k: v for k, v in model.state_dict().items() if k != '_metadata'}
    torch.save({
        'model_state_dict':     sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch':                epoch,
    }, path)


def fit(model, train_loader, val_loader,
        optimizer, scheduler, loss_fn, device, n_epochs, step_fn,
        *, start_epoch=0, prev_losses=None, monitor=None, metric_fn=None, log_fn=None,
        epoch_log_fn=None):
    """
    Loop completo de treino.

    Parameters
    ----------
    start_epoch : epoch absoluto inicial (0 = treino novo).
    prev_losses : {'train': [...], 'val': [...]} de run anterior; None = vazio.
    monitor     : TrainingMonitor | None — gerencia checkpoint, best e early stop.
    metric_fn   : (batch, model, device) -> (mae_hw, mae_graph|None) | None
                  Se fornecido, calcula MAE bruto sobre val_loader a cada heartbeat
                  (mesmo ritmo de checkpoint_every) e grava mae_hw/mae_graph no log
                  via monitor. mae_hw compara sempre a saída em grade H×W do FNO;
                  mae_graph compara a saída final em grafo/nós (None se o arch não
                  produzir saída em grafo).
    log_fn      : (ep, train_loss, val_loss, train_time_s, eval_time_s, samples_per_s) -> None | None
                  Print de época — propriedade da loss (BaseLoss.log_epoch,
                  src/neural_op/losses.py), passado por scripts/train.py como
                  `loss_obj.log_epoch`. None usa _default_log_epoch (mesmo texto
                  de sempre) — mantém fit() utilizável sem essa peça.
    epoch_log_fn: callable(**row) | None — chamado em TODA época
                  com epoch/train_loss/val_loss/lr (lr usado NA época, antes do
                  scheduler.step)/train_time_s/eval_time_s/metric_time_s/
                  epoch_time_s/mae_hw/mae_graph. mae_* só vêm preenchidos em toda
                  época se monitor.cfg.metrics_every_epoch=True (senão só nos
                  heartbeats, None nas demais). ModelManager.log_epoch -> epochs.csv.

    Returns
    -------
    model : nn.Module
    losses : {'train': list[float], 'val': list[float]}
    """
    log_fn = log_fn or _default_log_epoch
    model = model.to(device)
    # Após load_state_dict com map_location='cpu', os buffers de momentum do Adam
    # ficam na CPU enquanto os parâmetros já estão no device — move o estado junto.
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f"Modelo: {count_params_real(model):,} parâmetros reais  |  device: {device}")
    if start_epoch > 0:
        print(f"Retomando do epoch {start_epoch} — treinando {n_epochs} épocas adicionais")

    train_losses = list(prev_losses['train']) if prev_losses else []
    val_losses   = list(prev_losses['val'])   if prev_losses else []

    metrics_every_epoch = (metric_fn is not None and monitor is not None
                           and getattr(monitor.cfg, 'metrics_every_epoch', False))

    for i in range(n_epochs):
        ep = start_epoch + i
        epoch_lr = optimizer.param_groups[0]['lr']   # lr usado nesta época
        t0 = time.perf_counter()
        train_loss, n_train_samples = train_epoch(model, train_loader, optimizer, loss_fn, device, step_fn)
        t1 = time.perf_counter()
        val_loss = eval_epoch(model, val_loader, loss_fn, device, step_fn)
        t2 = time.perf_counter()
        is_heartbeat = monitor is not None and (i + 1) % monitor.cfg.checkpoint_every == 0
        # mae_hw/mae_graph: toda época (metrics_every_epoch) ou só no heartbeat;
        # calculado 1x e reaproveitado pelo monitor abaixo
        if metric_fn is not None and (metrics_every_epoch or is_heartbeat):
            mae_hw, mae_graph = compute_mae_metrics(model, val_loader, device, metric_fn)
        else:
            mae_hw, mae_graph = None, None
        t3 = time.perf_counter()
        if scheduler is not None:
            scheduler.step()
        train_time_s  = t1 - t0
        eval_time_s   = t2 - t1
        samples_per_s = n_train_samples / train_time_s if train_time_s > 0 else 0.0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        log_fn(ep, train_loss, val_loss, train_time_s, eval_time_s, samples_per_s)
        if epoch_log_fn is not None:
            epoch_log_fn(epoch=ep, train_loss=train_loss, val_loss=val_loss, lr=epoch_lr,
                         train_time_s=train_time_s, eval_time_s=eval_time_s,
                         metric_time_s=t3 - t2, epoch_time_s=t3 - t0,
                         mae_hw=mae_hw, mae_graph=mae_graph)

        if monitor is not None:
            monitor.last_epoch = ep
            if is_heartbeat:
                current_lr = optimizer.param_groups[0]['lr']
                should_stop = monitor.step(
                    ep, train_losses, val_losses, model, optimizer, scheduler,
                    lr=current_lr, train_time_s=train_time_s,
                    eval_time_s=eval_time_s, samples_per_s=samples_per_s,
                    mae_hw=mae_hw, mae_graph=mae_graph,
                )
                if should_stop:
                    break

    return model, {'train': train_losses, 'val': val_losses}
