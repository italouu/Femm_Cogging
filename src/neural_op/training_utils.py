import torch
import time
from functools import reduce
import operator
# [REMOVIDO] Path migrado para src/neural_op/monitor.py
# from pathlib import Path


# [REMOVIDO] mse_loss migrado para src/neural_op/losses.py — use LOSS_REGISTRY['mse']
# def mse_loss(out, y):
#     return torch.mean((out - y) ** 2)

# [REMOVIDO] l2_loss renomeado para mse_loss — nome anterior era impreciso (calculava MSE, não L2)
# def l2_loss(out, y):
#     l2 = torch.mean((out - y) ** 2)
#     return l2

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def _batch_size(batch):
    """Conta amostras no batch independente do formato (tuple grid ou dict qtree)."""
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
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            mae_hw, mae_graph = metric_fn(batch, model, device)
            total_hw += mae_hw
            if mae_graph is not None:
                total_graph += mae_graph
                has_graph = True
            n_batches += 1
    if n_batches == 0:
        return 0.0, None
    mae_graph_avg = (total_graph / n_batches) if has_graph else None
    return total_hw / n_batches, mae_graph_avg


def _default_log_epoch(ep, train_loss, test_loss, train_time_s, eval_time_s, samples_per_s):
    """log_epoch padrão -- usado se fit() for chamado sem log_fn (ex: scripts
    antigos/testes) ou por qualquer BaseLoss que não sobrescreva log_epoch
    (src/neural_op/losses.py). Mesmo texto que já era hardcoded aqui."""
    print(f"epoch {ep:>4d}  train {train_loss:.4e}  test {test_loss:.4e}"
          f"  [{train_time_s:.1f}s + {eval_time_s:.1f}s eval]  {samples_per_s:.0f} samp/s")


def save_checkpoint(path, epoch, model, optimizer, scheduler
                    # [REMOVIDO] train_losses, test_losses → metrics.jsonl via ModelManager
                    # [REMOVIDO] extra → config.json via ModelManager
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


# [REMOVIDO] TrainingMonitor migrado para src/neural_op/monitor.py
# class TrainingMonitor: ...


def fit(model, train_loader, test_loader,
        optimizer, scheduler, loss_fn, device, n_epochs, step_fn,
        *, start_epoch=0, prev_losses=None, monitor=None, metric_fn=None, log_fn=None):
    """
    Loop completo de treino.

    Parameters
    ----------
    start_epoch : epoch absoluto inicial (0 = treino novo).
    prev_losses : {'train': [...], 'test': [...]} de run anterior; None = vazio.
    monitor     : TrainingMonitor | None — gerencia checkpoint, best e early stop.
    metric_fn   : (batch, model, device) -> (mae_hw, mae_graph|None) | None
                  Se fornecido, calcula MAE bruto sobre test_loader a cada heartbeat
                  (mesmo ritmo de checkpoint_every) e grava mae_hw/mae_graph no log
                  via monitor. mae_hw compara sempre a saída em grade H×W do FNO;
                  mae_graph compara a saída final em grafo/nós (None se o arch não
                  produzir saída em grafo).
    log_fn      : (ep, train_loss, test_loss, train_time_s, eval_time_s, samples_per_s) -> None | None
                  Print de época — propriedade da loss (BaseLoss.log_epoch,
                  src/neural_op/losses.py), passado por scripts/train.py como
                  `loss_obj.log_epoch`. None usa _default_log_epoch (mesmo texto
                  de sempre) — mantém fit() utilizável sem essa peça.

    Returns
    -------
    model : nn.Module
    losses : {'train': list[float], 'test': list[float]}
    """
    # [REMOVIDO] checkpoint_every, checkpoint_path, ckpt_extra migrados para TrainingMonitor
    # checkpoint_every=0, checkpoint_path=None, ckpt_extra=None
    log_fn = log_fn or _default_log_epoch
    model = model.to(device)
    # Após load_state_dict com map_location='cpu', os buffers de momentum do Adam
    # ficam na CPU enquanto os parâmetros já estão no device — move o estado junto.
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f"Modelo: {count_params(model):,} parâmetros  |  device: {device}")
    if start_epoch > 0:
        print(f"Retomando do epoch {start_epoch} — treinando {n_epochs} épocas adicionais")

    train_losses = list(prev_losses['train']) if prev_losses else []
    test_losses  = list(prev_losses['test'])  if prev_losses else []

    for i in range(n_epochs):
        ep = start_epoch + i
        t0 = time.perf_counter()
        train_loss, n_train_samples = train_epoch(model, train_loader, optimizer, loss_fn, device, step_fn)
        t1 = time.perf_counter()
        test_loss = eval_epoch(model, test_loader, loss_fn, device, step_fn)
        t2 = time.perf_counter()
        if scheduler is not None:
            scheduler.step()
        train_time_s  = t1 - t0
        eval_time_s   = t2 - t1
        samples_per_s = n_train_samples / train_time_s if train_time_s > 0 else 0.0
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        # [REMOVIDO 2026-08-18] print hardcoded — virou log_fn (propriedade da
        # loss, ver BaseLoss.log_epoch em src/neural_op/losses.py), chamado
        # logo abaixo. _default_log_epoch imprime o mesmo texto de antes.
        # print(f"epoch {ep:>4d}  train {train_loss:.4e}  test {test_loss:.4e}"
        #       f"  [{train_time_s:.1f}s + {eval_time_s:.1f}s eval]  {samples_per_s:.0f} samp/s")
        log_fn(ep, train_loss, test_loss, train_time_s, eval_time_s, samples_per_s)

        # [REMOVIDO] bloco de checkpoint inline — substituído por TrainingMonitor
        # if checkpoint_every > 0 and checkpoint_path and (i + 1) % checkpoint_every == 0:
        #     save_checkpoint(checkpoint_path, ep, ...)

        if monitor is not None:
            monitor.last_epoch = ep
            if (i + 1) % monitor.cfg.checkpoint_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                mae_hw, mae_graph = (
                    compute_mae_metrics(model, test_loader, device, metric_fn)
                    if metric_fn is not None else (None, None)
                )
                should_stop = monitor.step(
                    ep, train_losses, test_losses, model, optimizer, scheduler,
                    lr=current_lr, train_time_s=train_time_s,
                    eval_time_s=eval_time_s, samples_per_s=samples_per_s,
                    mae_hw=mae_hw, mae_graph=mae_graph,
                )
                if should_stop:
                    break

    return model, {'train': train_losses, 'test': test_losses}


# [REMOVIDO] stack_to_bchw e bchw_to_stack descartados junto com Quad_FNO2d
#            (src/neural_op/quad_fno.py) — eram usados exclusivamente por ele.

# def stack_to_bchw(L, dim, cells, data):
#     """Transform a concatenated stream [1, C, S] into [B, C, H, W]."""
#     device = data.device
#     L     = torch.as_tensor(L, device=device, dtype=torch.long)
#     cells = torch.as_tensor(cells, device=device, dtype=torch.long)
#     B = int(L.numel())
#     C = int(data.size(1))
#     H, W = int(dim[0]), int(dim[1])
#     S = int(L.sum().item())
#     assert cells.numel() == S
#     block_id = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.long), L)
#     vals = data.squeeze(0).transpose(0, 1).contiguous()
#     out = torch.zeros(B, C, H*W, dtype=data.dtype, device=device)
#     ch_idx   = torch.arange(C, device=device, dtype=torch.long).view(1,-1).expand(S,-1)
#     blk_idx  = block_id.view(-1,1).expand(-1,C)
#     cell_idx = cells.view(-1,1).expand(-1,C)
#     out.index_put_((blk_idx, ch_idx, cell_idx), vals, accumulate=True)
#     return out.view(B, C, H, W)

# def bchw_to_stack(L, cells, conv_data, base_res):
#     """Rebuild the concatenated stream [1, C, S] from a BCHW tensor."""
#     device = conv_data.device
#     dtype  = conv_data.dtype
#     L     = torch.as_tensor(L, device=device, dtype=torch.long)
#     cells = torch.as_tensor(cells, device=device, dtype=torch.long)
#     B, C, Hc, Wc = conv_data.shape
#     H, W = int(base_res[0]), int(base_res[1])
#     block_id = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.long), L)
#     r = torch.div(cells, W, rounding_mode='floor')
#     c = torch.remainder(cells, W)
#     out_SC = conv_data[block_id, :, r, c]
#     out = out_SC.transpose(0, 1).unsqueeze(0).to(dtype)
#     return out
