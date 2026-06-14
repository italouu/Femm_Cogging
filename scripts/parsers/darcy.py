"""
darcy.py
--------
Converte o arquivo HDF5 do PDEBench (2D DarcyFlow beta=1.0) para
data_chunk_*.pt compatíveis com ChunkStreamDataset (mode='grid').

Uso (a partir da raiz do projeto):
    python -m scripts.parsers.darcy
    python -m scripts.parsers.darcy --file data/external/darcy/2D_DarcyFlow_beta1.0_Train.hdf5
    python -m scripts.parsers.darcy --file <caminho.hdf5> --chunk-size 32

Como obter o arquivo HDF5:
    https://github.com/pdebench/PDEBench  (seção "Download Data")
    Arquivo : 2D_DarcyFlow_beta1.0_Train.hdf5   (1000 amostras, arquivo único)
    URL     : https://darus.uni-stuttgart.de/api/access/datafile/133219
    Destino : data/external/darcy/

    Não há arquivo de teste separado no PDEBench — o split treino/teste é feito
    em runtime por build_loaders (NnCfg.train_split + split_seed).

Estrutura HDF5 real (confirmada pelo código PDEBench models/fno/utils.py):
    nu     : [N, H, W]     float32  — coeficiente de entrada (3D)
    tensor : [N, 1, H, W]  float32  — solução de saída (4D, T=1 steady-state)
    (chaves alternativas detectadas automaticamente)

Formato dos chunks gerados:
    x_hw : [B, 1, H, W]  float32  — coeficiente (canal único)
    y_hw : [B, 1, H, W]  float32  — solução (canal único)
    dim  : (H, W)
"""

import argparse
from pathlib import Path

import numpy as np
import torch

_ROOT      = Path(__file__).parents[2]
_EXT_DIR   = _ROOT / "data" / "external" / "darcy"
_CHUNK_DIR = _ROOT / "data" / "torch" / "data_chunks"

_X_KEYS = ('nu', 'input', 'x')
_Y_KEYS = ('tensor', 'output', 'y')


def _open_hdf5(path: Path):
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py não instalado. Execute: pip install h5py")
    return h5py.File(path, 'r')


def _detect_keys(f) -> tuple[str, str]:
    x_key = next((k for k in _X_KEYS if k in f), None)
    y_key = next((k for k in _Y_KEYS if k in f), None)
    if x_key is None or y_key is None:
        raise KeyError(
            f"Chaves esperadas não encontradas. Disponíveis: {list(f.keys())}.\n"
            f"  x esperado em: {_X_KEYS}\n"
            f"  y esperado em: {_Y_KEYS}"
        )
    return x_key, y_key


def _to_nhw(arr: np.ndarray, name: str) -> np.ndarray:
    """Garante shape [N, H, W] — remove dimensão T=1 se presente."""
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"Campo '{name}': esperado T=1 (steady-state), "
                f"obtido shape {arr.shape}"
            )
        return arr[:, 0, :, :]
    raise ValueError(f"Campo '{name}': shape inesperado {arr.shape} (esperado 3D ou 4D)")


def hdf5_to_chunks(hdf5_path: Path, chunk_dir: Path, chunk_size: int) -> int:
    """Lê um HDF5 do PDEBench e salva data_chunk_*.pt. Retorna número de chunks novos."""
    chunk_dir.mkdir(parents=True, exist_ok=True)

    with _open_hdf5(hdf5_path) as f:
        x_key, y_key = _detect_keys(f)
        print(f"  chaves detectadas: x='{x_key}'  y='{y_key}'")
        print(f"  shape bruto: x={tuple(f[x_key].shape)}  y={tuple(f[y_key].shape)}")
        x_all = _to_nhw(np.array(f[x_key], dtype=np.float32), x_key)   # [N, H, W]
        y_all = _to_nhw(np.array(f[y_key], dtype=np.float32), y_key)   # [N, H, W]

    if x_all.shape != y_all.shape:
        raise ValueError(f"Shape mismatch após normalização: x={x_all.shape} y={y_all.shape}")

    N, H, W = x_all.shape
    dim = (H, W)
    print(f"  {N} amostras  {H}x{W}")

    n_new = 0
    for i in range(0, N, chunk_size):
        chunk_idx = i // chunk_size
        out_path  = chunk_dir / f"data_chunk_{chunk_idx:04d}.pt"
        if out_path.exists():
            continue

        slc     = slice(i, min(i + chunk_size, N))
        x_chunk = torch.from_numpy(x_all[slc, None, :, :])   # [B, 1, H, W]
        y_chunk = torch.from_numpy(y_all[slc, None, :, :])   # [B, 1, H, W]

        torch.save({'x_hw': x_chunk, 'y_hw': y_chunk, 'dim': dim}, out_path)
        print(f"  [chunk {chunk_idx:04d}] {x_chunk.shape[0]} amostras")
        n_new += 1

    return n_new


def run(hdf5_path, chunk_size: int = 32, dataset: str = 'darcy_beta1') -> int:
    chunk_dir = _CHUNK_DIR / dataset

    print(f"\n=== Parser Darcy → {dataset} ===")
    print(f"  arquivo    : {Path(hdf5_path).name}")
    print(f"  chunk_size : {chunk_size}")
    print(f"  destino    : {chunk_dir}")
    print()

    n_new   = hdf5_to_chunks(Path(hdf5_path), chunk_dir, chunk_size)
    n_total = len(list(chunk_dir.glob("data_chunk_*.pt")))

    if n_new == 0:
        print("\n  Nenhum chunk novo gerado (todos já existem).")
    else:
        print(f"\n=== {n_new} chunk(s) novos  |  {n_total} total em {chunk_dir} ===")

    return n_new


def main():
    default_file = str(_EXT_DIR / '2D_DarcyFlow_beta1.0_Train.hdf5')

    ap = argparse.ArgumentParser(
        description='Converte DarcyFlow HDF5 (PDEBench) → data_chunk_*.pt'
    )
    ap.add_argument(
        '--file',
        default=default_file,
        help=f'Caminho para o HDF5 (padrão: {default_file})',
    )
    ap.add_argument('--chunk-size', type=int, default=32,
                    help='Amostras por chunk (padrão: 32)')
    ap.add_argument('--dataset', default='darcy_beta1',
                    help='Subdiretório em data/torch/quad_chunks/ (padrão: darcy_beta1)')
    args = ap.parse_args()

    hdf5_path = Path(args.file)
    if not hdf5_path.exists():
        print(f"Arquivo não encontrado: {hdf5_path}")
        print(f"\nBaixe o arquivo do PDEBench e coloque em: {_EXT_DIR}")
        print("  Arquivo : 2D_DarcyFlow_beta1.0_Train.hdf5")
        print("  URL     : https://darus.uni-stuttgart.de/api/access/datafile/133219")
        print("\nOu passe o caminho explicitamente:")
        print("  python -m scripts.parsers.darcy --file <caminho.hdf5>")
        return

    run(hdf5_path, args.chunk_size, args.dataset)


if __name__ == '__main__':
    main()
