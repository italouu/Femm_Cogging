#!/usr/bin/env bash
# run_vm_pipeline.sh — preparação da bateria definitiva mesh_ans_138x276_unified na VM (Linux).
#
# Rodar da RAIZ do projeto:
#   bash scripts/run_vm_pipeline.sh                 # pré-check + chunks + Fase 0 + V1/V2 + params + V3
#   bash scripts/run_vm_pipeline.sh --archive       # idem + B0 (move os logs antigos p/ ..._oldcriterion)
#   bash scripts/run_vm_pipeline.sh --archive --battery   # idem + dispara a bateria (run_best_configs)
#   bash scripts/run_vm_pipeline.sh --only phase0   # roda só uma etapa (ignora o marcador .done)
#
# Etapas (nesta ordem — a Fase 0 e o V2 leem os checkpoints/config.json das runs ANTIGAS,
# então precisam rodar ANTES do B0):
#   precheck  versão do Python, CUDA/GPU, disco, raw e logs antigos presentes
#   chunks    python -m scripts.build_unified_ans_chunks_direct   (~22 min, ~105 GB, retomável)
#   phase0    python -m scripts.phase0_measurements               (test set inteiro)
#   v1, v2    python -m scripts.verify_battery v1 / v2
#   params    python -m scripts.count_battery_params
#   v3        python -m scripts.verify_battery v3                 (smoke 4 archs × losses, 3 épocas)
#   archive   python -m scripts.archive_old_battery_logs --execute  [só com --archive]
#   battery   python -m scripts.run_best_configs                  [só com --battery; exige archive]
#
# Para na PRIMEIRA falha. Cada etapa concluída grava data/logs/_vm_pipeline/.done_<etapa>
# e é pulada numa nova execução (apague o marcador para refazer). Log de cada etapa em
# data/logs/_vm_pipeline/<etapa>.log.
#
# Variáveis: PYTHON (default python3), MIN_FREE_GB (default 110).

set -euo pipefail

PYTHON="${PYTHON:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-110}"
PROBLEM="mesh_ans_138x276_unified_best_mse_mae"
RAW_DIR="data/raw/mesh_ans_138x276"
CHUNK_ROOT="data/torch/data_chunks/mesh_ans_138x276_unified"
OLD_LOGS="data/logs/${PROBLEM}"
STATE_DIR="data/logs/_vm_pipeline"

DO_ARCHIVE=0
DO_BATTERY=0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive) DO_ARCHIVE=1 ;;
        --battery) DO_BATTERY=1 ;;
        --only)    ONLY="$2"; shift ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ ! -f "scripts/run_best_configs.py" ]]; then
    echo "ERRO: rode da raiz do projeto (scripts/run_best_configs.py não encontrado)" >&2
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

precheck() {
    local fail=0
    echo "--- git"
    git rev-parse --short HEAD || true
    if [[ -n "$(git status --porcelain -- src scripts 2>/dev/null)" ]]; then
        echo "AVISO: alterações não commitadas em src/ ou scripts/ (git_dirty=true nos logs)"
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
# GNN_PostBase (width 64 x 6 camadas, batch 32): pico ~25 GiB no smoke (RTX 4080 16 GB falhou)
if gib < 26:
    print('AVISO: GPU com < 26 GiB — GNN_PostBase (batch 32) pode estourar memória (pico ~25 GiB)')
import shapely, numpy, pandas, matplotlib, scipy  # dependências da cadeia de parsing/treino
EOF

    echo "--- disco"
    local free_gb
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    echo "livre: ${free_gb} GB"
    if [[ ! -f "$STATE_DIR/.done_chunks" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
        echo "ERRO: < ${MIN_FREE_GB} GB livres e os chunks (~105 GB) ainda não foram gerados"
        fail=1
    fi

    echo "--- raw"
    local n_raw
    n_raw=$(find "$RAW_DIR" -maxdepth 1 -name 'sample_*.ans.gz' 2>/dev/null | wc -l)
    echo "$RAW_DIR: $n_raw .ans.gz"
    [[ "$n_raw" -eq 4000 ]] || { echo "ERRO: esperado 4000 sample_*.ans.gz"; fail=1; }
    [[ -f "$RAW_DIR/valid_designs.csv" ]] || { echo "ERRO: falta valid_designs.csv"; fail=1; }

    echo "--- logs da bateria antiga (Fase 0 e V2 precisam deles)"
    if [[ -f "$STATE_DIR/.done_phase0" && -f "$STATE_DIR/.done_v2" ]]; then
        echo "Fase 0 e V2 já concluídas — logs antigos não são mais necessários"
    else
        local n_best
        n_best=$(find "$OLD_LOGS" -path '*/checkpoints/best.pth' 2>/dev/null | wc -l)
        echo "$OLD_LOGS: $n_best best.pth"
        [[ "$n_best" -ge 8 ]] || { echo "ERRO: esperado 8 runs (4 archs × mse/mae) com best.pth"; fail=1; }
    fi

    return $fail
}

run_step precheck  precheck
run_step chunks    "$PYTHON" -m scripts.build_unified_ans_chunks_direct
run_step phase0    "$PYTHON" -m scripts.phase0_measurements
run_step v1        "$PYTHON" -m scripts.verify_battery v1
run_step v2        "$PYTHON" -m scripts.verify_battery v2
run_step params    "$PYTHON" -m scripts.count_battery_params
run_step v3        "$PYTHON" -m scripts.verify_battery v3

if [[ "$DO_ARCHIVE" == 1 || "$ONLY" == "archive" ]]; then
    run_step archive "$PYTHON" -m scripts.archive_old_battery_logs --execute
fi

if [[ "$DO_BATTERY" == 1 || "$ONLY" == "battery" ]]; then
    if [[ ! -f "$STATE_DIR/.done_archive" ]]; then
        echo "ERRO: a bateria exige o B0 (archive) concluído — rode com --archive" >&2
        exit 1
    fi
    run_step battery "$PYTHON" -m scripts.run_best_configs
fi

echo
echo "Pipeline concluído. Resultados em docs/bateria_definitiva/ (phase0_results.*, verify_results.json,"
echo "param_counts_b3.json). Revisar antes de --archive/--battery."
