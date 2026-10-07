#!/usr/bin/env bash
# run_vm_elem_proto_chunks.sh — PROTÓTIPO FNO_BipartiteGNN_Elem (2026-10-07): DATASET na VM (Linux).
#
# Gera os chunks do protótipo (bipartite com papéis trocados: elementos = grafo principal com
# saída Bx,By por elemento; vértices = auxiliar só com posição) direto do raw da bateria —
# mesma referência: data/raw/mesh_ans_138x276/, 125 chunks × 32 na mesma ordem → mesmo split.
# Treino: scripts/run_vm_elem_proto_train.sh (depois deste).
#
# Rodar da RAIZ do projeto:
#   bash scripts/run_vm_elem_proto_chunks.sh
#   nohup bash scripts/run_vm_elem_proto_chunks.sh > data/logs/_vm_elem_proto/chunks.out 2>&1 &
#
# Etapas (para na primeira falha; marcador data/logs/_vm_elem_proto/.done_<etapa>; --only <etapa>):
#   precheck_chunks  Python, disco (~47 GB), raw
#   chunks           python -m scripts.build_elem_proto_chunks   (retomável — pula chunks existentes)
#
# Variáveis: PYTHON (default python3), MIN_FREE_GB (default 55).

set -euo pipefail

PYTHON="${PYTHON:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-55}"
RAW_DIR="data/raw/mesh_ans_138x276"
CHUNK_DIR="data/torch/data_chunks/mesh_ans_138x276_unified/FNO_BipartiteGNN_Elem"
STATE_DIR="data/logs/_vm_elem_proto"
N_CHUNKS=125

ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY="$2"; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ ! -f "scripts/build_elem_proto_chunks.py" ]]; then
    echo "ERRO: rode da raiz do projeto (scripts/build_elem_proto_chunks.py não encontrado)" >&2
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

precheck_chunks() {
    local fail=0
    echo "--- git"
    git rev-parse --short HEAD || true

    echo "--- python"
    "$PYTHON" -c "import sys, numpy, matplotlib, scipy, torch; print('python', sys.version.split()[0], '| torch', torch.__version__)" || fail=1

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

run_step precheck_chunks precheck_chunks
run_step chunks          "$PYTHON" -m scripts.build_elem_proto_chunks

echo
echo "Dataset pronto em $CHUNK_DIR. Próximo passo: bash scripts/run_vm_elem_proto_train.sh"
