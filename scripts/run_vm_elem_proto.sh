#!/usr/bin/env bash
# [REMOVIDO 2026-10-07] script único substituído, a pedido do usuário, por dois executáveis:
#   scripts/run_vm_elem_proto_chunks.sh  (dataset)
#   scripts/run_vm_elem_proto_train.sh   (smoke + treino)
# Corpo antigo mantido abaixo só para rastreabilidade — não executa.
echo "run_vm_elem_proto.sh descontinuado — use scripts/run_vm_elem_proto_chunks.sh e depois scripts/run_vm_elem_proto_train.sh" >&2
exit 2
# run_vm_elem_proto.sh — PROTÓTIPO FNO_BipartiteGNN_Elem (2026-10-07) na VM (Linux), ponta a ponta.
#
# Bipartite com os papéis trocados: elementos = grafo principal (saída Bx,By por elemento),
# vértices = auxiliar estático. Mesmo raw e mesma referência da bateria (mesh_ans_138x276,
# 125 chunks × 32 na mesma ordem → mesmo split, mesmos hiperparâmetros do FNO_BipartiteGNN).
#
# Rodar da RAIZ do projeto (de preferência em tmux/screen ou nohup):
#   bash scripts/run_vm_elem_proto.sh               # precheck + dataset + smoke + treino (mse, mae)
#   bash scripts/run_vm_elem_proto.sh --no-train    # só precheck + dataset + smoke
#   bash scripts/run_vm_elem_proto.sh --only <etapa>   # uma etapa (ignora o marcador .done)
#   nohup bash scripts/run_vm_elem_proto.sh > data/logs/_vm_elem_proto/pipeline.out 2>&1 &
#
# Etapas, nesta ordem (para na PRIMEIRA falha):
#   precheck  Python, CUDA/GPU, disco (~47 GB), raw
#   chunks    python -m scripts.build_elem_proto_chunks   (retomável — pula chunks existentes)
#   smoke     python -m tests.proto_elem_smoke --dataset <dataset completo> --train-epochs 2
#             (parser vs bateria, gradientes, treino de 2 épocas em data/logs/_smoke_elem_proto/)
#   train     python -m scripts.run_elem_proto            (mse e mae; logs em
#             data/logs/mesh_ans_138x276_unified_elem_proto/FNO_BipartiteGNN_Elem/)
#
# Cada etapa concluída grava data/logs/_vm_elem_proto/.done_<etapa> e é pulada numa nova
# execução (apague o marcador para refazer). Log de cada etapa em data/logs/_vm_elem_proto/<etapa>.log.
#
# Variáveis: PYTHON (default python3), MIN_FREE_GB (default 55).

set -euo pipefail

PYTHON="${PYTHON:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-55}"
RAW_DIR="data/raw/mesh_ans_138x276"
DATASET="mesh_ans_138x276_unified/FNO_BipartiteGNN_Elem"
CHUNK_DIR="data/torch/data_chunks/${DATASET}"
STATE_DIR="data/logs/_vm_elem_proto"
N_CHUNKS=125

DO_TRAIN=1
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-train) DO_TRAIN=0 ;;
        --only)     ONLY="$2"; shift ;;
        -h|--help)  sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ ! -f "scripts/run_elem_proto.py" ]]; then
    echo "ERRO: rode da raiz do projeto (scripts/run_elem_proto.py não encontrado)" >&2
    exit 2
fi
mkdir -p "$STATE_DIR"

# run_step <nome> <comando...> — pula se já concluída (exceto com --only), loga, marca .done
run_step() {
    local name="$1"; shift
    if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
        return 0
    fi
    if [[ -z "$ONLY" && -f "$STATE_DIR/.done_$name" ]]; then
        echo "[skip] $name (já concluída em $(cat "$STATE_DIR/.done_$name"))"
        return 0
    fi
    echo
    echo "======================================================================"
    echo "[run ] $name: $*"
    echo "       log: $STATE_DIR/$name.log"
    echo "======================================================================"
    local t0=$SECONDS
    if "$@" 2>&1 | tee "$STATE_DIR/$name.log"; then
        date '+%Y-%m-%dT%H:%M:%S' > "$STATE_DIR/.done_$name"
        echo "[ ok ] $name ($(( (SECONDS - t0) / 60 )) min)"
    else
        echo "[FAIL] $name — ver $STATE_DIR/$name.log" >&2
        exit 1
    fi
}

chunks_complete() {
    local n
    n=$(find "$CHUNK_DIR" -maxdepth 1 -name 'data_chunk_*.pt' 2>/dev/null | wc -l)
    [[ "$n" -ge "$N_CHUNKS" ]]
}

precheck() {
    local fail=0
    echo "--- git"
    git rev-parse --short HEAD || true
    if [[ -n "$(git status --porcelain -- src scripts tests 2>/dev/null)" ]]; then
        echo "AVISO: alterações não commitadas em src/, scripts/ ou tests/"
    fi

    echo "--- python / torch / GPU"
    "$PYTHON" - <<'EOF' || fail=1
import sys
print('python', sys.version.split()[0])
import torch
print('torch', torch.__version__)
if not torch.cuda.is_available():
    sys.exit('ERRO: CUDA indisponível')
p = torch.cuda.get_device_properties(0)
gib = p.total_memory / 2**30
print(f'GPU {p.name}  {gib:.1f} GiB')
# medido 2026-10-07 (RTX 4080): ~0,56 GiB/amostra no treino → batch 32 ≈ 18 GiB
if gib < 20:
    print('AVISO: GPU com < 20 GiB — batch 32 (~18 GiB estimados) pode estourar memória')
import numpy, matplotlib, scipy
EOF

    echo "--- disco"
    local free_gb
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    echo "livre: ${free_gb} GB"
    if chunks_complete; then
        echo "chunks já completos ($N_CHUNKS) — requisito de disco não se aplica"
    elif [[ "$free_gb" -lt "$MIN_FREE_GB" ]]; then
        echo "ERRO: < ${MIN_FREE_GB} GB livres e os chunks (~47 GB) ainda não foram gerados"
        fail=1
    fi

    echo "--- raw"
    local n_raw
    n_raw=$(find "$RAW_DIR" -maxdepth 1 -name 'sample_*.ans.gz' 2>/dev/null | wc -l)
    echo "$RAW_DIR: $n_raw .ans.gz"
    [[ "$n_raw" -eq 4000 ]] || { echo "ERRO: esperado 4000 sample_*.ans.gz"; fail=1; }
    [[ -f "$RAW_DIR/valid_designs.csv" ]] || { echo "ERRO: falta valid_designs.csv"; fail=1; }

    return $fail
}

run_step precheck precheck
run_step chunks   "$PYTHON" -m scripts.build_elem_proto_chunks
run_step smoke    "$PYTHON" -m tests.proto_elem_smoke --dataset "$DATASET" --train-epochs 2

if [[ "$DO_TRAIN" == 1 || "$ONLY" == "train" ]]; then
    run_step train "$PYTHON" -m scripts.run_elem_proto
    echo
    echo "Treino concluído. Logs em data/logs/mesh_ans_138x276_unified_elem_proto/FNO_BipartiteGNN_Elem/."
else
    echo
    echo "Dataset e smoke prontos (--no-train). Para treinar: bash scripts/run_vm_elem_proto.sh"
fi
