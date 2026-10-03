#!/usr/bin/env bash
# run_vm_pipeline.sh — bateria definitiva mesh_ans_138x276_unified na VM (Linux), ponta a ponta.
#
# Rodar da RAIZ do projeto (de preferência dentro de tmux/screen ou com nohup — a bateria
# leva horas e cai se a sessão da VDI desconectar):
#   scripts/run_vm_pipeline.sh                  # TUDO: verificações + B0 + bateria
#   scripts/run_vm_pipeline.sh --no-battery     # tudo menos a bateria (para revisar antes)
#   scripts/run_vm_pipeline.sh --only <etapa>   # roda só uma etapa (ignora o marcador .done)
#   nohup scripts/run_vm_pipeline.sh > data/logs/_vm_pipeline/pipeline.out 2>&1 &
#
# Etapas, nesta ordem (para na PRIMEIRA falha — a bateria só roda se o smoke V3 passar):
#   precheck  Python, CUDA/GPU, disco, raw; avisa se os logs antigos não estão presentes
#   chunks    python -m scripts.build_unified_ans_chunks_direct   (pula chunks já existentes)
#   phase0    python -m scripts.phase0_measurements               (test set inteiro)
#             [só se os 8 best.pth da bateria antiga estiverem em data/logs/<PROBLEM>/;
#              senão é pulada com AVISO e sem marcador — pode ser rodada depois]
#   v1, v2    python -m scripts.verify_battery v1 / v2
#   params    python -m scripts.count_battery_params
#   v3        python -m scripts.verify_battery v3                 (smoke 4 archs × losses, 3 épocas)
#   archive   B0: python -m scripts.archive_old_battery_logs --execute
#             [só se data/logs/<PROBLEM>/ tiver runs; sem Fase 0 usa --skip-phase0-check]
#   battery   python -m scripts.run_best_configs  (B1–B4 ligados; B5/B6 pelas chaves
#             N_REPEATS/INCLUDE_REL_L2 no topo de scripts/run_best_configs.py)
#
# Cada etapa concluída grava data/logs/_vm_pipeline/.done_<etapa> e é pulada numa nova
# execução (apague o marcador para refazer). Log de cada etapa em data/logs/_vm_pipeline/<etapa>.log.
# ATENÇÃO: a etapa battery não é retomável por run — se falhar no meio, rodar de novo
# recomeça as 8 runs (as já feitas ficam em run_XXXX anteriores).
#
# Variáveis: PYTHON (default python3), MIN_FREE_GB (default 110).

# [REMOVIDO 2026-10-03] cabeçalho da versão em etapas (B0/bateria só com --archive/--battery)
# — substituído pela execução ponta a ponta, a pedido do usuário (um único script).
# # run_vm_pipeline.sh — preparação da bateria definitiva mesh_ans_138x276_unified na VM (Linux).
# #
# # Rodar da RAIZ do projeto:
# #   bash scripts/run_vm_pipeline.sh                 # pré-check + chunks + Fase 0 + V1/V2 + params + V3
# #   bash scripts/run_vm_pipeline.sh --archive       # idem + B0 (move os logs antigos p/ ..._oldcriterion)
# #   bash scripts/run_vm_pipeline.sh --archive --battery   # idem + dispara a bateria (run_best_configs)
# #   bash scripts/run_vm_pipeline.sh --only phase0   # roda só uma etapa (ignora o marcador .done)
# #
# # Etapas (nesta ordem — a Fase 0 e o V2 leem os checkpoints/config.json das runs ANTIGAS,
# # então precisam rodar ANTES do B0):
# #   precheck  versão do Python, CUDA/GPU, disco, raw e logs antigos presentes
# #   chunks    python -m scripts.build_unified_ans_chunks_direct   (~22 min, ~105 GB, retomável)
# #   phase0    python -m scripts.phase0_measurements               (test set inteiro)
# #   v1, v2    python -m scripts.verify_battery v1 / v2
# #   params    python -m scripts.count_battery_params
# #   v3        python -m scripts.verify_battery v3                 (smoke 4 archs × losses, 3 épocas)
# #   archive   python -m scripts.archive_old_battery_logs --execute  [só com --archive]
# #   battery   python -m scripts.run_best_configs                  [só com --battery; exige archive]
# #
# # Para na PRIMEIRA falha. Cada etapa concluída grava data/logs/_vm_pipeline/.done_<etapa>
# # e é pulada numa nova execução (apague o marcador para refazer). Log de cada etapa em
# # data/logs/_vm_pipeline/<etapa>.log.
# #
# # Variáveis: PYTHON (default python3), MIN_FREE_GB (default 110).

set -euo pipefail

PYTHON="${PYTHON:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-110}"
PROBLEM="mesh_ans_138x276_unified_best_mse_mae"
RAW_DIR="data/raw/mesh_ans_138x276"
CHUNK_ROOT="data/torch/data_chunks/mesh_ans_138x276_unified"
OLD_LOGS="data/logs/${PROBLEM}"
STATE_DIR="data/logs/_vm_pipeline"

DO_BATTERY=1
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        # [REMOVIDO] --archive/--battery como opt-in — agora são o padrão (aceitos como no-op)
        # --archive) DO_ARCHIVE=1 ;;
        # --battery) DO_BATTERY=1 ;;
        --archive|--battery) ;;
        --no-battery) DO_BATTERY=0 ;;
        --only)    ONLY="$2"; shift ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
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

# n_old_best — nº de best.pth da bateria antiga (Fase 0 precisa dos 8)
n_old_best() {
    find "$OLD_LOGS" -path '*/checkpoints/best.pth' 2>/dev/null | wc -l
}

# chunks_complete — 125 chunks em cada uma das 4 pastas do dataset unificado
chunks_complete() {
    local a n
    for a in FNO2d FNO_GNN GNN_PostBase FNO_BipartiteGNN; do
        n=$(find "$CHUNK_ROOT/$a" -maxdepth 1 -name 'data_chunk_*.pt' 2>/dev/null | wc -l)
        [[ "$n" -ge 125 ]] || return 1
    done
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
# [REMOVIDO] shapely/pandas não são importados por nenhum script da cadeia (verificado
# importando todos e inspecionando sys.modules) — o precheck falhava à toa na VM sem shapely.
# import shapely, numpy, pandas, matplotlib, scipy  # dependências da cadeia de parsing/treino
import numpy, matplotlib, scipy  # dependências reais da cadeia de parsing/treino/eval
EOF

    echo "--- disco"
    local free_gb
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    echo "livre: ${free_gb} GB"
    if chunks_complete; then
        echo "chunks unificados completos (4 × 125) — requisito de disco dos chunks não se aplica"
    fi
    # [REMOVIDO] exigia espaço livre mesmo com os chunks já presentes
    # if [[ ! -f "$STATE_DIR/.done_chunks" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    if [[ ! -f "$STATE_DIR/.done_chunks" ]] && ! chunks_complete && [[ "$free_gb" -lt "$MIN_FREE_GB" ]]; then
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
        n_best=$(n_old_best)
        echo "$OLD_LOGS: $n_best best.pth"
        # [REMOVIDO] logs antigos eram obrigatórios — agora só a Fase 0 depende deles e é pulada
        # [[ "$n_best" -ge 8 ]] || { echo "ERRO: esperado 8 runs (4 archs × mse/mae) com best.pth"; fail=1; }
        if [[ "$n_best" -lt 8 ]]; then
            echo "AVISO: sem os 8 best.pth da bateria antiga — a Fase 0 (T0a/T0b/T0c) será PULADA."
            echo "       Para incluí-la: copiar config.json/split.json/checkpoints/best.pth das 8 runs"
            echo "       para $OLD_LOGS/ ANTES de rodar este script."
        fi
    fi

    return $fail
}

# [REMOVIDO 2026-10-03] sequência da versão em etapas (B0/bateria opt-in) — ver cabeçalho
# run_step precheck  precheck
# run_step chunks    "$PYTHON" -m scripts.build_unified_ans_chunks_direct
# run_step phase0    "$PYTHON" -m scripts.phase0_measurements
# run_step v1        "$PYTHON" -m scripts.verify_battery v1
# run_step v2        "$PYTHON" -m scripts.verify_battery v2
# run_step params    "$PYTHON" -m scripts.count_battery_params
# run_step v3        "$PYTHON" -m scripts.verify_battery v3
#
# if [[ "$DO_ARCHIVE" == 1 || "$ONLY" == "archive" ]]; then
#     run_step archive "$PYTHON" -m scripts.archive_old_battery_logs --execute
# fi
#
# if [[ "$DO_BATTERY" == 1 || "$ONLY" == "battery" ]]; then
#     if [[ ! -f "$STATE_DIR/.done_archive" ]]; then
#         echo "ERRO: a bateria exige o B0 (archive) concluído — rode com --archive" >&2
#         exit 1
#     fi
#     run_step battery "$PYTHON" -m scripts.run_best_configs
# fi
#
# echo
# echo "Pipeline concluído. Resultados em docs/bateria_definitiva/ (phase0_results.*, verify_results.json,"
# echo "param_counts_b3.json). Revisar antes de --archive/--battery."

run_step precheck  precheck
run_step chunks    "$PYTHON" -m scripts.build_unified_ans_chunks_direct

if [[ -n "$ONLY" || -f "$STATE_DIR/.done_phase0" || "$(n_old_best)" -ge 8 ]]; then
    run_step phase0 "$PYTHON" -m scripts.phase0_measurements
else
    echo "[WARN] phase0 PULADA — logs antigos ausentes (sem marcador .done; pode rodar depois)"
fi

run_step v1        "$PYTHON" -m scripts.verify_battery v1
run_step v2        "$PYTHON" -m scripts.verify_battery v2
run_step params    "$PYTHON" -m scripts.count_battery_params
run_step v3        "$PYTHON" -m scripts.verify_battery v3

# B0 — arquiva a bateria antiga (se houver runs) para a nova começar com a pasta vazia
if [[ -n "$(find "$OLD_LOGS" -mindepth 2 -maxdepth 2 -name 'run_*' 2>/dev/null | head -1)" ]]; then
    if [[ -f "$STATE_DIR/.done_phase0" ]]; then
        run_step archive "$PYTHON" -m scripts.archive_old_battery_logs --execute
    else
        run_step archive "$PYTHON" -m scripts.archive_old_battery_logs --execute --skip-phase0-check
    fi
elif [[ -z "$ONLY" ]]; then
    echo "[skip] archive — $OLD_LOGS sem runs (nada a arquivar)"
fi

if [[ "$DO_BATTERY" == 1 ]]; then
    run_step battery "$PYTHON" -m scripts.run_best_configs
    echo
    echo "Bateria concluída. Logs em $OLD_LOGS/<arch>/run_XXXX/ (config.json, epochs.csv,"
    echo "run_summary.json, checkpoints/best.pth, model_final.pth)."
else
    echo
    echo "Pipeline concluído sem a bateria (--no-battery). Resultados em docs/bateria_definitiva/."
    echo "Para disparar: scripts/run_vm_pipeline.sh  (etapas concluídas são puladas)"
fi
