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

    # npz_parser: chave em PARSER_REGISTRY. Nos quatro modos, consumido por
    # gen_npz_structures.py (mesmo comando `python -m scripts.gen_npz_structures`
    # pros quatro modos):
    #   mode='grid'/'qtree'  -> filtra colunas na criação do .npz a partir das
    #                           CSVs de data/raw/<dataset>/
    #   mode='femm_mesh'     -> branch próprio em gen_npz_structures.py, filtra
    #                           colunas do staging bruto já gerado em
    #                           data/raw/<dataset>/ (sample_*.npz) — usar
    #                           'FEMM_MESH' ou variante futura
    #   mode='femm_mesh_v2'  -> branch próprio (_run_femm_mesh_v2), só usa
    #                           cfg.target_field do parser (as demais colunas
    #                           não se aplicam — ver 'FEMM_MESH_V2'/
    #                           'FEMM_MESH_V2_A' na tabela abaixo)
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
    #   FEMM_MESH_V2    | FEMM_MESH_V2_PARSER   | mode='femm_mesh_v2' — grafo duplo
    #                   |                       | vértices+elementos, alvo B (Bx,By
    #                   |                       | via curl(A)) — arch FNO_BipartiteGNN
    #   FEMM_MESH_V2_A  | FEMM_MESH_V2_A_PARSER | mode='femm_mesh_v2' — mesma entrada,
    #                   |                       | alvo escalar A
    #
    # FEMM_MESH/FEMM_MESH_A só são compatíveis com mode='femm_mesh';
    # FEMM_MESH_V2/FEMM_MESH_V2_A só com mode='femm_mesh_v2'; os demais são
    # do pipeline CSV (grid/qtree).
    #
    # [ATUALIZADO 2026-08-19] mode='femm_mesh_v2' passou a usar PARSER_REGISTRY
    # (antes era ignorado; a escolha do alvo vinha de um campo dedicado,
    # femm_mesh_v2_target_field, descontinuado — ver [REMOVIDO] abaixo).
    # gen_npz_structures.py::_run_femm_mesh_v2 lê só
    # PARSER_REGISTRY[npz_parser].target_field e repassa pra
    # src/data_gen/parsers/femm_mesh_v2.py::parse_ans_gzip_sample, que já
    # produz o formato final (grafo de vértices [node_x=r,c / node_y=A ou
    # Bx,By] + grafo de elementos [elem_x=mu_r,M,area,r,c] + arestas cruzadas
    # + grade H×W) -- não há seleção de colunas a fazer, por isso
    # FEMM_MESH_V2_PARSER/FEMM_MESH_V2_A_PARSER só carregam target_field (ver
    # docstring desses módulos). Arch correspondente: FNO_BipartiteGNN
    # (src/configs/training.py). Ver CLAUDE.md, "Grafo duplo vértices+elementos"
    # (2026-08-10).
    npz_parser:             str           = 'FEMM_MESH_V2'
    npz_samples_per_worker: int           = 2
    npz_max_workers:        int           = 6
    npz_max_samples:        Optional[int] = None

    # generate_data_femm_mesh (só mode='femm_mesh')
    femm_mesh_max_workers: int = 8

    # [REMOVIDO 2026-08-19] target_field de mode='femm_mesh_v2' descontinuado
    # como campo dedicado — agora vem de PARSER_REGISTRY[npz_parser].target_field
    # (ver npz_parser acima e femm_mesh_v2_dataset_name abaixo), mesmo
    # mecanismo usado pelos demais modos. Motivo: reduzir a assimetria de
    # mode='femm_mesh_v2' não seguir o contrato de parser já estabelecido
    # pelo resto do projeto (PARSER_REGISTRY/DatagenConfig.npz_parser).
    # femm_mesh_v2_target_field: str = 'B'

    @property
    def femm_mesh_v2_dataset_name(self) -> str:
        """Nome do subdiretório usado por mode='femm_mesh_v2' em
        data/temp/samples_npz/ e data/torch/data_chunks/ (gen_npz_structures.py/
        build_data_chunks_femm_mesh_v2.py). Sem sufixo quando target_field='A'
        (mantém compatibilidade com os 4000 .npz já processados em produção,
        ver CLAUDE.md "Pendências conhecidas"); sufixado com '_B' só quando
        target_field='B', pra não misturar/colidir com o staging de A já
        existente (o raw .ans.gz continua o mesmo -- só o parse difere).

        target_field agora vem de PARSER_REGISTRY[self.npz_parser] (2026-08-19)
        -- import local (não no topo do módulo) pelo mesmo motivo de
        NnCfg.__post_init__ (src/configs/training.py) importar ARCH_REGISTRY
        dentro do método: evita puxar a cadeia de imports de
        src/data_gen/parsers/ (matplotlib.tri, scipy) pra todo consumidor de
        DatagenConfig, mesmo quem só precisa dos campos simples."""
        from src.data_gen.parsers import PARSER_REGISTRY
        target_field = PARSER_REGISTRY[self.npz_parser].target_field
        if target_field == 'A':
            return self.dataset
        return f"{self.dataset}_{target_field}"

    @property
    def parsed_dataset_name(self) -> str:
        """Nome combinado dataset+parser (mode='femm_mesh') — usado como
        subdiretório em data/temp/samples_mesh_parsed/ e
        data/torch/data_chunks/, pra não misturar chunks de parsers
        diferentes (o layout de node_x/edge_attr muda conforme o parser)."""
        return f"{self.dataset}_{self.npz_parser}"
