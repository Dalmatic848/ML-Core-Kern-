#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_overnight.sh — последовательное ночное обучение нескольких конфигураций
#
# Запуск:
#   bash run_overnight.sh             # все эксперименты
#   bash run_overnight.sh --skip 2    # пропустить первые 2
#   bash run_overnight.sh --only 3    # только эксперимент №3
#
# Структура каждого эксперимента:
#   1. python3 train.py ...          — обучение
#   2. python3 visualize_results.py  — графики на тесте
#
# Логи: logs/overnight_YYYY-MM-DD_HH-MM/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"
source ../venv/bin/activate

# ── Параметры ─────────────────────────────────────────────────────────────────
SKIP_N=0
ONLY_N=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip) SKIP_N=$2; shift 2 ;;
        --only) ONLY_N=$2; shift 2 ;;
        *) echo "Неизвестный аргумент: $1"; exit 1 ;;
    esac
done

# ── Папка для логов ───────────────────────────────────────────────────────────
LOG_DIR="logs/overnight_$(date +%Y-%m-%d_%H-%M)"
mkdir -p "$LOG_DIR"
echo "Логи: $LOG_DIR"
echo "Старт: $(date)" | tee "$LOG_DIR/summary.txt"

# ── Вспомогательные функции ───────────────────────────────────────────────────

run_experiment() {
    local idx=$1
    local name=$2
    local train_cmd=$3
    local vis_cmd=$4

    # Фильтрация --skip / --only
    if [[ -n "$ONLY_N" && "$idx" != "$ONLY_N" ]]; then return; fi
    if [[ "$idx" -le "$SKIP_N" ]]; then
        echo "⏭  [${idx}] ${name} — пропущен (--skip ${SKIP_N})"
        return
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  [${idx}/${TOTAL}] ${name}"
    echo "  Старт: $(date)"
    echo "════════════════════════════════════════════════════════════════"

    local log_train="${LOG_DIR}/${idx}_${name}_train.log"
    local log_vis="${LOG_DIR}/${idx}_${name}_vis.log"

    # Обучение
    echo "  → train..."
    if eval "$train_cmd" 2>&1 | tee "$log_train" | \
        grep --line-buffered -E "^Ep |early stop|Ctrl\+C|Error|Traceback" ; then
        echo "  ✓ train завершён: $(date)" | tee -a "$LOG_DIR/summary.txt"
    else
        echo "  ✗ train ОШИБКА (код $?): $(date)" | tee -a "$LOG_DIR/summary.txt"
        return
    fi

    # Визуализация
    echo "  → visualize..."
    if eval "$vis_cmd" 2>&1 | tee "$log_vis" | \
        grep --line-buffered -E "Сохранено|Готово|macro|F1=|Error|Traceback" ; then
        echo "  ✓ vis завершена: $(date)" | tee -a "$LOG_DIR/summary.txt"
    else
        echo "  ✗ vis ОШИБКА: $(date)" | tee -a "$LOG_DIR/summary.txt"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Список экспериментов
# ─────────────────────────────────────────────────────────────────────────────
#
# Общие флаги:
#   --scheduler plateau   → ReduceLROnPlateau (не осциллирует как OneCycleLR)
#   --max-epochs 40       → ограничение чтобы влезть в ночь
#   --patience 10         → быстрее останавливается если нет прогресса
#
# Датасеты:
#   data/dataset_random   → 6 классов, случайный сплит
#   data/dataset_soft     → 30 минералов → 5 базовых компонент (soft labels)

COMMON_TRAIN="--scheduler plateau --max-epochs 40 --patience 10"
DATASET_6CLS="--dataset-root data/dataset_random"
DATASET_SOFT="--dataset-root data/dataset_soft"

TOTAL=5

# [1] ResNet18 + soft labels + plateau
# Самый быстрый. Тестируем soft+plateau — лучшее из обоих миров
run_experiment 1 "rn18_soft_plateau" \
    "python3 train.py run_soft_plateau --mode dual --soft-labels $DATASET_SOFT $COMMON_TRAIN" \
    "python3 visualize_soft.py --run run_soft_plateau --no-columns"

# [2] ResNet50 + 6 классов + plateau
# Сравниваем RN50 vs RN18 при правильном планировщике
run_experiment 2 "rn50_plateau" \
    "python3 train.py run_rn50_plateau --mode dual --arch resnet50 $DATASET_6CLS $COMMON_TRAIN" \
    "python3 visualize_results.py run_rn50_plateau --dual $DATASET_6CLS --no-photos"

# [3] EfficientNet-B3 + 6 классов + plateau
# Современная архитектура, меньше параметров чем RN50, возможно лучше
run_experiment 3 "effb3_plateau" \
    "python3 train.py run_effb3_plateau --mode dual --arch efficientnet_b3 $DATASET_6CLS $COMMON_TRAIN" \
    "python3 visualize_results.py run_effb3_plateau --dual $DATASET_6CLS --no-photos"

# [4] ConvNeXt-Tiny + 6 классов + plateau
# Современная сверточная архитектура (2022), часто превосходит ResNet
run_experiment 4 "convnext_plateau" \
    "python3 train.py run_convnext_plateau --mode dual --arch convnext_tiny $DATASET_6CLS $COMMON_TRAIN" \
    "python3 visualize_results.py run_convnext_plateau --dual $DATASET_6CLS --no-photos"

# [5] ResNet50 + soft labels + plateau
# Мощная модель + правильные метки — финальный эксперимент если успеет
run_experiment 5 "rn50_soft_plateau" \
    "python3 train.py run_rn50_soft_plateau --mode dual --arch resnet50 --soft-labels $DATASET_SOFT $COMMON_TRAIN" \
    "python3 visualize_soft.py --run run_rn50_soft_plateau --no-columns"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Все эксперименты завершены: $(date)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Итоговые метрики:"
python3 - << 'PYEOF'
import json
from pathlib import Path

results_root = Path('results')
for run_name in ['run_soft_plateau', 'run_rn50_plateau', 'run_effb3_plateau',
                 'run_convnext_plateau', 'run_rn50_soft_plateau']:
    p = results_root / run_name / 'metrics_summary.json'
    if not p.exists():
        print(f"  {run_name}: нет metrics_summary.json")
        continue
    data = json.load(open(p))
    best = max(data, key=lambda x: x['val_f1'])
    print(f"  {run_name}: val_f1={best['val_f1']:.4f} (эп {best['epoch']})")
PYEOF
echo ""
echo "Логи: $LOG_DIR"
echo "Конец: $(date)" | tee -a "$LOG_DIR/summary.txt"
