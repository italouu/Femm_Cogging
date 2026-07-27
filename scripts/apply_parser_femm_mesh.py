# [REMOVIDO] 2026-07-27 — lógica migrada para o branch mode='femm_mesh' de
# scripts/gen_npz_structures.py (_run_femm_mesh/_load_npz_femm_mesh/
# _apply_parser_femm_mesh), pra manter o mesmo comando/etapa do pipeline
# qtree/grid (python -m scripts.gen_npz_structures) em vez de um script
# próprio separado. Único ajuste de conteúdo na migração: origem passou de
# data/temp/samples_mesh/<dataset>/ para data/raw/<dataset>/ (staging bruto
# não é dado descartável — ver nota em generate_data_femm_mesh.py) e destino
# de data/temp/samples_mesh_parsed/<dataset>_<npz_parser>/ para
# data/temp/samples_npz/<dataset>_<npz_parser>/ (mesma pasta-base usada pelo
# pipeline qtree/grid). Arquivo mantido comentado por rastreabilidade —
# não apagar (ver CLAUDE.md).
#
# """
# apply_parser_femm_mesh.py
# --------------------------
# Etapa intermediária do pipeline mode='femm_mesh': converte o .npz de staging
# BRUTO (data/temp/samples_mesh/<dataset>/, gerado por
# generate_data_femm_mesh.py — TODOS os dados extraídos da malha) no .npz
# FINAL (data/temp/samples_mesh_parsed/<dataset>_<npz_parser>/ — só o que o
# treino consome, já no formato definitivo). Duas responsabilidades, ambas só
# aqui: (1) aplicar o parser (PARSER_REGISTRY, seleção de colunas de
# node_x/node_y/edge_attr) e (2) descartar metadados que só existem pra
# depuração/rastreabilidade do bruto (ver _DROP_KEYS). Mesmo papel que
# gen_npz_structures.py cumpre no pipeline qtree, mas em etapa própria: aqui o
# staging bruto já existe (custou uma simulação FEM inteira por amostra) e não
# deve ser regenerado só porque o parser mudou — só essa etapa roda de novo.
# build_data_chunks_femm_mesh.py, a jusante, é concatenação pura — não filtra
# nem descarta nada por conta própria.
#
# NÃO reaproveita apply_parser_config (src/data_gen/parsers/_base.py): aquela
# função não filtra edge_attr (no pipeline qtree essa seleção acontece em
# build_graph_edges_motor, na criação do .npz — papel que aqui cabe a este
# script).
#
# Execução (a partir da raiz do projeto):
#     python -m scripts.apply_parser_femm_mesh
# """
# from pathlib import Path
#
# import numpy as np
#
# from src.configs.datagen import DatagenConfig
# from src.data_gen.parsers import PARSER_REGISTRY, MotorQtreeParserConfig
#
# _dg        = DatagenConfig()
# DATASET    = _dg.dataset
# NPZ_PARSER = _dg.npz_parser
# PARSER     = PARSER_REGISTRY[NPZ_PARSER]
#
# # None = processa todas as amostras disponíveis
# MAX_SAMPLES = None
#
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# _STAGING_DIR = PROJECT_ROOT / "data" / "temp" / "samples_mesh" / DATASET
# _OUT_DIR     = PROJECT_ROOT / "data" / "temp" / "samples_mesh_parsed" / _dg.parsed_dataset_name
#
# # Metadados de geometria por amostra do staging BRUTO — só rastreabilidade/
# # depuração (ver src/data_gen/femm_mesh.generate_mesh_sample), não é dado
# # final de treino. Este é o único lugar que decide isso: build_data_chunks_
# # femm_mesh.py é concatenação pura, não descarta mais nada por conta própria.
# _DROP_KEYS = frozenset({'r_in_mm', 'r_ext_mm', 'ang_1_deg', 'ang_2_deg'})
#
#
# def _load_npz(path: Path) -> dict:
#     return {key: v for key, v in np.load(path).items() if key not in _DROP_KEYS}
#
#
# def _apply_parser(d: dict, cfg: MotorQtreeParserConfig) -> dict:
#     """Filtra node_x/node_y/x_hw/y_hw/edge_attr pelas colunas do parser.
#     Demais chaves (node_A, a_hw, edge_index, L, dim_H/W) passam inalteradas
#     — não têm variante "cheia" a filtrar."""
#     out = dict(d)
#     out['node_x'] = d['node_x'][:, cfg.node_x_cols]
#     out['node_y'] = d['node_y'][:, cfg.node_y_cols]
#     out['x_hw']   = d['x_hw'][cfg.x_hw_cols]
#     out['y_hw']   = d['y_hw'][cfg.y_hw_cols]
#     if cfg.build_graph:
#         out['edge_attr'] = d['edge_attr'][:, cfg.edge_attr_cols]
#     else:
#         out.pop('edge_index', None)
#         out.pop('edge_attr', None)
#     return out
#
#
# def run(max_samples=MAX_SAMPLES):
#     _OUT_DIR.mkdir(parents=True, exist_ok=True)
#
#     npz_paths = sorted(_STAGING_DIR.glob("sample_*.npz"),
#                         key=lambda p: int(p.stem.split('_')[1]))
#     if max_samples is not None:
#         npz_paths = npz_paths[:max_samples]
#
#     total = len(npz_paths)
#     if total == 0:
#         print("Nenhum .npz encontrado em", _STAGING_DIR)
#         return 0
#
#     pending = [p for p in npz_paths if not (_OUT_DIR / p.name).exists()]
#     ja_prontos = total - len(pending)
#
#     print(f"\n=== Parser '{NPZ_PARSER}' sobre staging femm_mesh ===")
#     print(f"  origem  : {_STAGING_DIR}")
#     print(f"  destino : {_OUT_DIR}")
#     print(f"  Já processados: {ja_prontos}  |  A processar: {len(pending)}")
#
#     if not pending:
#         print("Nada a fazer.")
#         return 0
#
#     ok = 0
#     for path in pending:
#         d = _load_npz(path)
#         out = _apply_parser(d, PARSER)
#
#         # escrita atômica: salva em .tmp.npz e renomeia — mesma técnica de
#         # generate_mesh_sample/process_and_save_sample
#         stem = path.stem
#         out_path = _OUT_DIR / path.name
#         tmp_path = _OUT_DIR / f"{stem}.tmp"       # np.savez adiciona .npz -> .tmp.npz
#         tmp_npz  = _OUT_DIR / f"{stem}.tmp.npz"
#         np.savez(tmp_path, **out)
#         tmp_npz.replace(out_path)
#         ok += 1
#
#     print(f"\n=== Resumo apply_parser_femm_mesh ===")
#     print(f"  Salvos     : {ok}")
#     print(f"  Já prontos : {ja_prontos}")
#     return ok
#
#
# def main():
#     run()
#
#
# if __name__ == "__main__":
#     main()
