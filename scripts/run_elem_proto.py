"""
run_elem_proto.py
-----------------
PROTÓTIPO (2026-10-07) -- treina FNO_BipartiteGNN_Elem (bipartite com os
papéis trocados, saída Bx,By por elemento) com os MESMOS hiperparâmetros e
chaves de correção (B1-B4, N_EPOCHS) do FNO_BipartiteGNN da bateria
definitiva -- importados de scripts/run_best_configs.py, não copiados.

Dataset: data/torch/data_chunks/mesh_ans_138x276_unified/FNO_BipartiteGNN_Elem/
(python -m scripts.build_elem_proto_chunks). Mesmo split da bateria.
Logs em data/logs/mesh_ans_138x276_unified_elem_proto/FNO_BipartiteGNN_Elem/
(pasta separada da bateria -- é só um teste).

Atenção: mae_graph aqui é MAE por ELEMENTO, não comparável diretamente com o
mae_graph por nó da bateria.

Executável 2 de 2 do protótipo (o 1º é scripts/build_elem_proto_chunks.py).
Antes de treinar: precheck (CUDA/GPU, 125 chunks presentes) e smoke de 1
batch real (forward/backward com gradiente finito e não-nulo em F1, F2, lift
e FNO; metric_fn finito) -- pular com --no-precheck. Treina LOSSES em
sequência (mse, depois mae); --losses escolhe outras.

Execução (raiz do projeto; Windows ou VM Linux):
    python -m scripts.run_elem_proto
    python -m scripts.run_elem_proto --losses mae
"""
import argparse
import glob
import traceback

from src.configs.training import NnCfg, FNO_BipartiteGNN_ElemConfig
from src.configs.monitor import MonitorCfg
from scripts.run_best_configs import (
    UNIFIED_ROOT, INTERP_MODE, FNO_NODE_RESCALE, MODES1, MODES2, DATA_RES_BIPARTITE,
    N_EPOCHS, METRICS_EVERY_EPOCH,
)
from scripts.train import run

ARCH     = 'FNO_BipartiteGNN_Elem'
DATASET  = f'{UNIFIED_ROOT}/{ARCH}'
PROBLEM  = f'{UNIFIED_ROOT}_elem_proto'
LOSSES   = ['mse', 'mae']
# [REMOVIDO 2026-10-07] auxiliar com FNO@vértices — decisão do usuário: só posição
# AUX_FNO  = True          # False = troca "pura" (vértices só com posição)
AUX_FNO  = False         # auxiliar (vértices) só com posição [r_base, c_base]


def make_arch_cfg() -> FNO_BipartiteGNN_ElemConfig:
    # mesmos valores do FNO_BipartiteGNN em run_best_configs.BEST_CONFIGS
    return FNO_BipartiteGNN_ElemConfig(
        fno_modes1=MODES1, fno_modes2=MODES2, fno_conv_width=6, fno_conv_layers=4,
        fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64, fno_proj_layers=3,
        data_res=DATA_RES_BIPARTITE, gnn_node_width=32, gnn_n_layers=3, lambda_loss=0,
        interp_mode=INTERP_MODE, fno_node_rescale=FNO_NODE_RESCALE, aux_fno=AUX_FNO,
    )


def make_nn_cfg(loss: str, dataset: str = DATASET, problem: str = PROBLEM, **overrides) -> NnCfg:
    kw = dict(
        dataset=dataset, problem=problem, arch=ARCH, loss=loss,
        lr=0.01, scheduler_gamma=0.6,          # = FNO_BipartiteGNN da bateria
        arch_cfg=make_arch_cfg(),
        n_epochs=N_EPOCHS,
        monitor_cfg=MonitorCfg(metrics_every_epoch=METRICS_EVERY_EPOCH),
    )
    kw.update(overrides)
    return NnCfg(**kw)


N_CHUNKS_EXPECTED = 125


def precheck(dataset: str = DATASET, n_chunks_expected: int = N_CHUNKS_EXPECTED) -> bool:
    """GPU + chunks + smoke de 1 batch (batch 2) com o modelo real."""
    import math
    import torch
    from src.neural_op.archs import ARCH_REGISTRY
    from src.neural_op.dataloaders.grid_loader import build_loaders, CUDAPrefetcher
    from src.neural_op.losses import LOSS_REGISTRY
    from src.neural_op.normalization import Normalizer

    print("\n=== Precheck (treino) ===", flush=True)
    if not torch.cuda.is_available():
        print("  ERRO: CUDA indisponível")
        return False
    p = torch.cuda.get_device_properties(0)
    gib = p.total_memory / 2**30
    print(f"  GPU    : {p.name}  {gib:.1f} GiB")
    if gib < 20:   # medido 2026-10-07: ~0,56 GiB/amostra -> batch 32 ~ 18 GiB
        print("  AVISO: < 20 GiB — batch 32 (~18 GiB estimados) pode estourar memória")
    paths = sorted(glob.glob(f'data/torch/data_chunks/{dataset}/data_chunk_*.pt'))
    print(f"  chunks : {len(paths)} em data/torch/data_chunks/{dataset}/")
    if len(paths) < n_chunks_expected:
        print(f"  ERRO: esperado {n_chunks_expected} — rode antes: python -m scripts.build_elem_proto_chunks")
        return False

    nn = make_nn_cfg('mse', dataset=dataset)
    ac, entry = nn.arch_cfg, ARCH_REGISTRY[ARCH]
    norm = Normalizer.from_dict(nn.norm_stats)
    tl, *_ = build_loaders(paths[:2], batch_size=2, train_split=0.5, buffer_size=4,
                           num_workers=0, prefetch_factor=None, seed=nn.split_seed,
                           mode=entry.loader_mode)
    batch = next(iter(CUDAPrefetcher(tl, 'cuda', normalizer=norm)))
    model = entry.make_model(ac).to('cuda')
    model.normalizer = norm
    loss = entry.make_step_fn(ac, nn.loss_cfg)(batch, model, LOSS_REGISTRY['mse'], 'cuda')
    loss.backward()
    g = model.gnn
    checks = {
        f'dims node_in={ac.node_in_ch} elem_in={ac.elem_in_ch} edge={ac.edge_dim} out={ac.grid_out_ch}':
            (ac.node_in_ch, ac.elem_in_ch, ac.edge_dim, ac.grid_out_ch) == (5, 2, 3, 2),
        f'loss finita ({loss.item():.4f})': bool(torch.isfinite(loss).item()),
    }
    for name, w in [('F1 elem-elem', g.layers[0].msg_vertex.W.weight),
                    ('F2 vértice->elem', g.layers[0].msg_elem.W.weight),
                    ('lift', g.lift.weight), ('FNO', next(model.fno.parameters()))]:
        gr = w.grad
        checks[f'gradiente finito e não-nulo em {name}'] = (
            gr is not None and bool(torch.isfinite(gr).all().item()) and gr.abs().sum().item() > 0)
    mae_hw, mae_graph = entry.metric_fn(ac)(batch, model, 'cuda')
    checks[f'metric_fn mae_hw={mae_hw:.4f} T mae_graph(elem)={mae_graph:.4f} T'] = (
        math.isfinite(mae_hw) and math.isfinite(mae_graph))
    ok = True
    for msg, c in checks.items():
        print(f"  [{'ok' if c else 'FALHA'}] {msg}")
        ok &= bool(c)
    del model, batch
    torch.cuda.empty_cache()
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--losses', nargs='+', default=LOSSES)
    ap.add_argument('--no-precheck', action='store_true')
    a = ap.parse_args()
    if not a.no_precheck and not precheck():
        print("\nPrecheck falhou — nenhum treino iniciado.")
        raise SystemExit(2)

    summary = []
    # [REMOVIDO 2026-10-07] laço fixo em LOSSES — agora --losses (default LOSSES)
    # for loss in LOSSES:
    for loss in a.losses:
        label = f"{ARCH} / {DATASET} / loss={loss}"
        print(f"\n{'=' * 80}\n{label}\n{'=' * 80}", flush=True)
        try:
            summary.append((label, run(make_nn_cfg(loss)), None))
        except Exception as e:
            traceback.print_exc()
            summary.append((label, 'failed', str(e)))

    print(f"\n{'=' * 80}\nResumo\n{'=' * 80}")
    for label, status, err in summary:
        print(f"  [{status:8s}] {label}" + (f"  -- {err}" if err else ""))
    raise SystemExit(0 if all(s in ('done', 'stopped') for _, s, _ in summary) else 1)
