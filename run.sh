#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Полный пайплайн: подготовка данных → обучение → визуализация
#
# Использование:
#   ./run.sh                                    # ResNet18, run_v5
#   ./run.sh run_v6                             # ResNet18, своё имя
#   ./run.sh run_rn50_v1 resnet50               # ResNet50
#   ./run.sh run_rn50_dual resnet50 dual        # ResNet50, dual stream
#   ./run.sh run_warm resnet50 dual run_rn50_v1 # dual с тёплым стартом
#
# Ctrl+C завершает текущий шаг — графики всё равно строятся,
# скрипт переходит к следующему шагу.
# ─────────────────────────────────────────────────────────────────────────────

RUN_NAME="${1:-run_v5}"
ARCH="${2:-resnet18}"
MODE="${3:-single}"
WARM_START="${4:-}"

PYTHON="$(dirname "$0")/venv/bin/python3"
SCRIPT_DIR="$(dirname "$0")"

# ── Родительский шелл игнорирует Ctrl+C ──────────────────────────────────────
# Ctrl+C идёт только в текущий дочерний процесс (Python-скрипт).
# После его завершения шелл продолжает со следующего шага.
trap '' INT

# ── Вспомогательные функции ──────────────────────────────────────────────────
step() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  $*"
    echo "══════════════════════════════════════════════════════════"
}

run() {
    # Запускает команду, возвращает 0 даже если прервана Ctrl+C (exit 130)
    "$@"
    local code=$?
    if [ $code -eq 130 ]; then
        echo "  [прервано Ctrl+C, продолжаем...]"
    fi
    return 0
}

# ── Шаг 1: Подготовка данных ─────────────────────────────────────────────────
step "Шаг 1/4: Подготовка данных"
if [ -f "$SCRIPT_DIR/data/dataset/normalization_stats.json" ]; then
    echo "  Датасет уже готов — пропускаем"
else
    run "$PYTHON" "$SCRIPT_DIR/prepare_data.py"
fi

# ── Шаг 2 и 3: Обучение ──────────────────────────────────────────────────────
if [ "$MODE" = "dual" ]; then
    step "Шаг 2/4: Обучение Dual ($ARCH) → $RUN_NAME"
    WARM_ARG=""
    if [ -n "$WARM_START" ]; then
        WARM_ARG="--warm-start $SCRIPT_DIR/results/$WARM_START"
    fi
    run "$PYTHON" "$SCRIPT_DIR/train.py" "$RUN_NAME" \
        --arch "$ARCH" --mode dual $WARM_ARG

else
    step "Шаг 2/4: Обучение ДС ($ARCH) → $RUN_NAME"
    run "$PYTHON" "$SCRIPT_DIR/train.py" "$RUN_NAME" \
        --arch "$ARCH" --mode single --modality ДС

    step "Шаг 3/4: Обучение УФ ($ARCH) → $RUN_NAME"
    run "$PYTHON" "$SCRIPT_DIR/train.py" "$RUN_NAME" \
        --arch "$ARCH" --mode single --modality УФ
fi

# ── Шаг 4: Визуализация ──────────────────────────────────────────────────────
step "Шаг 4/4: Визуализация → $RUN_NAME"
DUAL_FLAG=""
if [ "$MODE" = "dual" ]; then DUAL_FLAG="--dual"; fi
run "$PYTHON" "$SCRIPT_DIR/visualize_results.py" "$RUN_NAME" \
    $DUAL_FLAG --no-photos

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Готово: results/$RUN_NAME"
echo "══════════════════════════════════════════════════════════"
