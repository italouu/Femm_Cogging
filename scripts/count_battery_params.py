"""
count_battery_params.py — contagem de parâmetros das 4 archs da bateria
definitiva, ANTES (configuração da bateria anterior) e DEPOIS do B3 (data_res =
grade real 138×276, modes = espectro completo sem sobreposição).

Não precisa de chunks: dimensões de entrada fixas do dataset unificado
(x_hw/y_hw 2 canais; FNO_GNN/GNN_PostBase node_x 5, edge_attr 4; Bipartite
node_x 2, edge_attr 3, elem_x 5, cross_edge_attr 1).

    python -m scripts.count_battery_params      # imprime tabela markdown + JSON

Colunas: numel (convenção antiga de count_params — complexo conta 1, é o
n_params dos config.json antigos), reais (complexo conta 2) e treináveis (reais;
no GNN_PostBase só o GNN novo — a base FNO2d é congelada). Também reporta
quantos parâmetros espectrais ficam "mortos" (sobrescritos pela sobreposição
weights1/weights2, sem gradiente) em cada configuração.
"""
import json
import warnings
from dataclasses import asdict
from pathlib import Path

from src.configs.training import (
    FNOConfig, FNO_GNNConfig, FNO_BipartiteGNNConfig, _from_dict_generic, GNN_PostBaseConfig,
)
from src.neural_op.archs import ARCH_REGISTRY
from src.neural_op.archs.fno import full_spectrum_modes
from src.neural_op.training_utils import count_params, count_params_real

GRID_HW = (138, 276)


def _fno_kw(modes1, modes2, data_res, width):
    return dict(modes1=modes1, modes2=modes2, conv_width=width, conv_layers=4,
                lift_width=64, lift_layers=3, proj_width=64, proj_layers=3, data_res=data_res)


def _dead_spectral(model_fno, H=GRID_HW[0]):
    """Parâmetros reais de weights1 sobrescritos por weights2 (linhas sobrepostas)."""
    m1 = model_fno.modes1
    overlap = max(0, 2 * m1 - H)
    if overlap == 0:
        return 0
    dead = 0
    for sc in model_fno.conv_layer.spec_convs:
        w = sc.weights1                         # [in, out, m1, m2] complexo
        dead += w.shape[0] * w.shape[1] * overlap * w.shape[3] * 2
    return dead


def build(arch, variant):
    if variant == 'antes':
        m1 = m2 = 270
        dr = (138, 276) if arch == 'FNO_BipartiteGNN' else (135, 270)
    else:
        m1, m2 = full_spectrum_modes(*GRID_HW)
        dr = GRID_HW

    if arch == 'FNO2d':
        cfg = FNOConfig(**_fno_kw(m1, m2, dr, 8))
        cfg.in_channels, cfg.out_channels = 2, 2
        model = ARCH_REGISTRY['FNO2d'].make_model(cfg)
        return model, model
    if arch == 'FNO_GNN':
        cfg = FNO_GNNConfig(fno_modes1=m1, fno_modes2=m2, fno_conv_width=6, fno_conv_layers=4,
                            fno_lift_width=64, fno_lift_layers=3, fno_proj_width=64,
                            fno_proj_layers=3, data_res=dr, gnn_node_width=32, gnn_n_layers=3)
        cfg.edge_dim, cfg.grid_in_ch, cfg.grid_out_ch, cfg.node_in_ch = 4, 2, 2, 5
        model = ARCH_REGISTRY['FNO_GNN'].make_model(cfg)
        return model, model.fno
    if arch == 'FNO_BipartiteGNN':
        cfg = FNO_BipartiteGNNConfig(fno_modes1=m1, fno_modes2=m2, fno_conv_width=6,
                                     fno_conv_layers=4, fno_lift_width=64, fno_lift_layers=3,
                                     fno_proj_width=64, fno_proj_layers=3, data_res=dr,
                                     gnn_node_width=32, gnn_n_layers=3)
        cfg.edge_dim, cfg.grid_in_ch, cfg.grid_out_ch, cfg.node_in_ch = 3, 2, 2, 2
        cfg.elem_in_ch, cfg.cross_edge_dim = 5, 1
        model = ARCH_REGISTRY['FNO_BipartiteGNN'].make_model(cfg)
        return model, model.fno
    if arch == 'GNN_PostBase':
        base_cfg = FNOConfig(**_fno_kw(m1, m2, dr, 8))
        base_cfg.in_channels, base_cfg.out_channels = 2, 2
        cfg = _from_dict_generic(GNN_PostBaseConfig, dict(
            base_run_dir='__inexistente__', base_checkpoint='best',
            gnn_node_width=64, gnn_n_layers=6, base_arch='FNO2d',
            base_arch_cfg=asdict(base_cfg), edge_dim=4, node_in_ch=5, base_out_ch=2))
        model = ARCH_REGISTRY['GNN_PostBase'].make_model(cfg)
        return model, model.base_model
    raise KeyError(arch)


def main():
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import contextlib, io
        for arch in ('FNO2d', 'FNO_GNN', 'GNN_PostBase', 'FNO_BipartiteGNN'):
            for variant in ('antes', 'depois'):
                with contextlib.redirect_stdout(io.StringIO()):   # print do fallback de base
                    model, fno = build(arch, variant)
                rows.append(dict(
                    arch=arch, variant=variant,
                    modes=(fno.modes1, fno.modes2), data_res=tuple(fno.data_res),
                    numel=count_params(model),
                    real=count_params_real(model),
                    trainable_real=count_params_real(model, trainable_only=True),
                    dead_spectral_real=_dead_spectral(fno),
                ))
    print('| arch | config | modes (efetivos) | data_res | numel (antigo) | reais | treináveis (reais) | espectrais mortos (reais) |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['arch']} | {r['variant']} | {r['modes'][0]}×{r['modes'][1]} | "
              f"{r['data_res'][0]}×{r['data_res'][1]} | {r['numel']:,} | {r['real']:,} | "
              f"{r['trainable_real']:,} | {r['dead_spectral_real']:,} |")
    out = Path('docs/bateria_definitiva/param_counts_b3.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'\nsalvo em {out}')


if __name__ == '__main__':
    main()
