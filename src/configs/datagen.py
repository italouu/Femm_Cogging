from dataclasses import dataclass
from typing import Optional


@dataclass
class DatagenConfig:
    dataset: str = 'mesh_ans_138x276'

    # Grade / Geometria
    n_r: int = 138
    n_a: int = 276
    ang_1: int = 0
    ang_2: int = 120

    # Geração de dados
    # 'grid'/'qtree'    -> pipeline CSV (generate_data.py + gen_npz_structures.py);
    # 'femm_mesh'       -> pipeline via malha real do FEMM (generate_data_femm_mesh.py),
    #                      grafo extraído do .ans do solver em vez de quadtree Shapely
    #                      (ver src/data_gen/femm_mesh.py). Não usa check_data/generate_one_batch
    #                      (esses só aceitam 'grid'/'qtree').
    # 'femm_mesh_v2'    -> só a etapa de geração BRUTA (generate_data_femm_mesh_v2.py):
    #                      desenha+malha+mi_analyze() e salva o .ans INTEIRO (texto puro do
    #                      FEMM, sem nenhuma extração/derivação) comprimido em gzip direto em
    #                      data/raw/<dataset>/sample_XXXXXX.ans.gz — sem mi_loadsolution()/
    #                      COM de pós-processamento (não precisa, só copia o arquivo que
    #                      mi_analyze() já escreveu em disco). Corrige a lacuna encontrada em
    #                      2026-08-07 (ver CLAUDE.md, seção "Extração de dados direto do
    #                      arquivo .ans"): o modo 'femm_mesh' descartava a malha/elementos
    #                      originais depois de derivar node_x/edge_index/etc., inviabilizando
    #                      recálculos futuros (ex: B exato por elemento) sem resimular. Etapas
    #                      de parser/chunk (gen_npz_structures.py/build_data_chunks.py) que lêem
    #                      esse .ans.gz ainda não existem — ver "Pendências conhecidas".
    mode: str = 'femm_mesh_v2'
    distribution: str = 'uniform'
    sample_method: str = 'legacy'  # 'fixed_geometry' |'constrained' | 'legacy' | 'constrained_lhs'
    n_samples: int = 4000
    max_depth: int = 1
    datagen_seed: int = 12
    cascade_buffer: Optional[int] = 1
    homogeneity_threshold: float = 0.90

    # Prepare / chunks
    chunk_size: int = 32

    # npz_parser: chave em PARSER_REGISTRY. Em ambos os modos, consumido por
    # gen_npz_structures.py (mesmo comando `python -m scripts.gen_npz_structures`
    # pros três modos):
    #   mode='grid'/'qtree'  -> filtra colunas na criação do .npz a partir das
    #                           CSVs de data/raw/<dataset>/
    #   mode='femm_mesh'     -> branch próprio em gen_npz_structures.py, filtra
    #                           colunas do staging bruto já gerado em
    #                           data/raw/<dataset>/ (sample_*.npz) — usar
    #                           'FEMM_MESH' ou variante futura
    #
    # Parsers disponíveis (PARSER_REGISTRY, src/data_gen/parsers/__init__.py):
    #   Chave           | Parser                | Uso
    #   ----------------|-----------------------|---------------------------------------
    #   FNO_GNN         | FNO_GNN_PARSER        | pipeline qtree — arch FNO_GNN
    #   FNO_GNN_v2      | FNO_GNN_V2_PARSER     | idem + delta_mu direcional em edge_attr
    #                   |                       | ([E,5]) — arch FNO_GNN_v2
    #   FNO2D           | FNO2D_PARSER          | pipeline grid — arch FNO2d
    #   MaskedFNO2d     | MASKED_FNO2D_PARSER   | pipeline grid — arch MaskedFNO2d
    #   MaskedFNO_GNN   | MASKED_FNO_GNN_PARSER | pipeline qtree — arch MaskedFNO_GNN
    #   FEMM_MESH       | FEMM_MESH_PARSER      | mode='femm_mesh' — malha real do FEMM,
    #                   |                       | alvo B (Bx,By)
    #   FEMM_MESH_A     | FEMM_MESH_A_PARSER    | mode='femm_mesh' — mesma entrada,
    #                   |                       | alvo escalar A
    #
    # Só FEMM_MESH/FEMM_MESH_A são compatíveis com mode='femm_mesh'; os demais
    # são do pipeline CSV (grid/qtree).
    #
    # mode='femm_mesh_v2' NÃO usa PARSER_REGISTRY -- npz_parser é ignorado nesse
    # modo. gen_npz_structures.py::_run_femm_mesh_v2 chama
    # src/data_gen/femm_mesh_v2.py::parse_ans_gzip_sample direto sobre o
    # sample_*.ans.gz bruto, que já produz o formato final (grafo de vértices
    # [node_x=r,c / node_y=A] + grafo de elementos [elem_x=mu_r,M,area,r,c] +
    # arestas cruzadas + grade H×W) -- não há seleção de colunas a fazer. Arch
    # correspondente: FNO_BipartiteGNN (src/configs/training.py). Ver CLAUDE.md,
    # "Grafo duplo vértices+elementos" (2026-08-10).
    npz_parser:             str           = 'FEMM_MESH_A'
    npz_samples_per_worker: int           = 2
    npz_max_workers:        int           = 12
    npz_max_samples:        Optional[int] = None

    # generate_data_femm_mesh (só mode='femm_mesh')
    femm_mesh_max_workers: int = 8

    # target_field p/ mode='femm_mesh_v2' (2026-08-13, src/data_gen/parsers/femm_mesh_v2.py::
    # parse_ans_gzip_sample) -- 'A' (padrão, potencial vetor escalar nos vértices, já em
    # produção pras 4000 amostras de mesh_ans_138x276) ou 'B' (Bx,By -- curl(A) fechado por
    # elemento, coordenadas reais em metros, + média nodal dos elementos incidentes; mesmo
    # padrão de amostragem de A na grade H×W, ver docstring do módulo). Ignorado nos demais
    # mode. Não afeta a etapa raw (.ans.gz é o mesmo .ans bruto independente do alvo
    # escolhido no parse) -- só data/temp/samples_npz/ e data/torch/data_chunks/, via
    # femm_mesh_v2_dataset_name abaixo.
    femm_mesh_v2_target_field: str = 'B'

    @property
    def femm_mesh_v2_dataset_name(self) -> str:
        """Nome do subdiretório usado por mode='femm_mesh_v2' em
        data/temp/samples_npz/ e data/torch/data_chunks/ (gen_npz_structures.py/
        build_data_chunks_femm_mesh_v2.py). Sem sufixo quando target_field='A'
        (mantém compatibilidade com os 4000 .npz já processados em produção,
        ver CLAUDE.md "Pendências conhecidas"); sufixado com '_B' só quando
        target_field='B', pra não misturar/colidir com o staging de A já
        existente (o raw .ans.gz continua o mesmo -- só o parse difere)."""
        if self.femm_mesh_v2_target_field == 'A':
            return self.dataset
        return f"{self.dataset}_{self.femm_mesh_v2_target_field}"

    @property
    def parsed_dataset_name(self) -> str:
        """Nome combinado dataset+parser (mode='femm_mesh') — usado como
        subdiretório em data/temp/samples_mesh_parsed/ e
        data/torch/data_chunks/, pra não misturar chunks de parsers
        diferentes (o layout de node_x/edge_attr muda conforme o parser)."""
        return f"{self.dataset}_{self.npz_parser}"
