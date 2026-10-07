#!/usr/bin/env bash
# [REMOVIDO 2026-10-07] substituído, a pedido do usuário, por dois executáveis Python:
#   python -m scripts.build_elem_proto_chunks   (dataset, com precheck)
#   python -m scripts.run_elem_proto            (precheck + smoke + treino)
# Corpo antigo mantido abaixo só para rastreabilidade — não executa.
echo "descontinuado — use python -m scripts.build_elem_proto_chunks e depois python -m scripts.run_elem_proto" >&2
exit 2
# run_vm_elem_proto_train.sh — PROTÓTIPO FNO_BipartiteGNN_Elem (2026-10-07): TREINO na VM (Linux).
#
# Exige os chunks prontos (bash scripts/run_vm_elem_proto_chunks.sh). Mesmos hiperparâmetros do
# FNO_BipartiteGNN da bateria (importados de scripts/run_best_configs.py); treina loss='mse' e
# depois loss='mae' (LOSSES no topo de scripts/run_elem_proto.py), em sequência.
# Logs: data/logs/mesh_ans_138x276_unified_elem_proto/FNO_BipartiteGNN_Elem/run_XXXX/.
#
# Rodar da RAIZ do projeto (de preferência em tmux/screen ou nohup):
#   bash scripts/run_vm_elem_proto_train.sh
#   nohup bash scripts/run_vm_elem_proto_train.sh > data/logs/_vm_elem_proto/train.out 2>&1 &
#
# Etapas (para na primeira falha; marcador data/logs/_vm_elem_proto/.done_<etapa>; --only <etapa>):
#   precheck_train  CUDA/GPU, chunks completos
#   smoke           python -m tests.proto_elem_smoke --dataset <dataset> --train-epochs 2
#                   (parser vs bateria, gradientes, treino de 2 épocas em data/logs/_smoke_elem_proto/)
#   train           python -m scripts.run_elem_proto   (mse, depois mae)
#
# Variáveis: PYTHON (default python3).

set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATASET="mesh_ans_138x276_unified/FNO_BipartiteGNN_Elem"
CHUNK_DIR="data/torch/data_chunks/${DATASET}"
STATE_DIR="data/logs/_vm_elem_proto"
N_CHUNKS=125

ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY="$2"; shift ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
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

precheck_train() {
    local fail=0
    echo "--- git"
    git rev-parse --short HEAD || true
    if [[ -n "$(git status --porcelain -- src scripts 2>/dev/null)" ]]; then
        echo "AVISO: alterações não commitadas em src/ ou scripts/"
    fi

    echo "--- python / torch / GPU"
    # medido 2026-10-07 (RTX 4080): ~0,56 GiB/amostra no treino -> batch 32 ~ 18 GiB
    "$PYTHON" -c "
import sys, torch
print('python', sys.version.split()[0], '| torch', torch.__version__)
if not torch.cuda.is_available():
    sys.exit('ERRO: CUDA indisponível')
p = torch.cuda.get_device_properties(0)
gib = p.total_memory / 2**30
print(f'GPU {p.name}  {gib:.1f} GiB')
if gib < 20:
    print('AVISO: GPU com < 20 GiB - batch 32 (~18 GiB estimados) pode estourar memória')
" || fail=1

    echo "--- chunks"
    local n
    n=$(find "$CHUNK_DIR" -maxdepth 1 -name 'data_chunk_*.pt' 2>/dev/null | wc -l)
    echo "$CHUNK_DIR: $n chunk(s)"
    if [[ "$n" -lt "$N_CHUNKS" ]]; then
        echo "ERRO: esperado $N_CHUNKS chunks — rode antes: bash scripts/run_vm_elem_proto_chunks.sh"
        fail=1
    fi
    if [[ ! -f "tests/proto_elem_smoke.py" ]]; then
        echo "ERRO: tests/proto_elem_smoke.py ausente (tests/ está no .gitignore — copiar para a VM)"
        fail=1
    fi

    return $fail
}

run_step precheck_train precheck_train
run_step smoke          "$PYTHON" -m tests.proto_elem_smoke --dataset "$DATASET" --train-epochs 2
run_step train          "$PYTHON" -m scripts.run_elem_proto

echo
echo "Treino concluído. Logs em data/logs/mesh_ans_138x276_unified_elem_proto/FNO_BipartiteGNN_Elem/."
