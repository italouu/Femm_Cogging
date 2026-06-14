"""
sample_processor.py
-------------------
Módulo responsável por processar e salvar uma única amostra do pipeline
quadtree. Ponto central de processamento: recebe um índice, executa tudo
(leitura de CSVs → unificação de quadtree → grafo → grade) e salva o .npz.

Uso:
    from src.data_gen.sample_processor import process_and_save_sample
    paths, falhas = process_and_save_sample(indices=[0, 1], unifier=unifier, out_dir=out_dir)
"""

from pathlib import Path


def process_and_save_sample(indices: list, unifier, out_dir) -> tuple:
    """
    Processa as amostras em `indices` e salva cada resultado em out_dir.

    O unifier retorna numpy puro — nenhum import de torch ocorre aqui.
    Escrita atômica: salva em .npz.tmp e renomeia para .npz ao final de cada
    amostra, evitando arquivos corrompidos em caso de falha.

    Parâmetros
    ----------
    indices : lista de índices a processar
    unifier : QtreeSampleUnifier já instanciado
    out_dir : diretório de saída (Path ou str)

    Retorna
    -------
    paths  : list[Path]             arquivos .npz salvos com sucesso
    falhas : list[tuple[int, str]]  (idx, mensagem de erro) por amostra que falhou
    """
    import numpy as np

    out_dir = Path(out_dir)
    paths  = []
    falhas = []

    for idx in indices:
        out_path = out_dir / f"sample_{idx:06d}.npz"
        tmp_path = out_dir / f"sample_{idx:06d}.tmp"      # np.savez adiciona .npz → .tmp.npz
        tmp_npz  = out_dir / f"sample_{idx:06d}.tmp.npz"
        try:
            sample = unifier[idx]
            # 'dim' é uma tupla — salva como dois escalares separados
            arrays = {k: v for k, v in sample.items() if k != 'dim'}
            arrays['dim_H'] = np.array(sample['dim'][0], dtype=np.int64)
            arrays['dim_W'] = np.array(sample['dim'][1], dtype=np.int64)
            np.savez(tmp_path, **arrays)   # grava em sample_{idx:06d}.tmp.npz
            tmp_npz.replace(out_path)      # atômico no Windows; substitui se já existir
            paths.append(out_path)
        except Exception as e:
            if tmp_npz.exists():
                tmp_npz.unlink()
            falhas.append((idx, repr(e)))

    return paths, falhas
