#!/usr/bin/env python3
"""
Пакетный ночной прогон нескольких экспериментов — замена run_overnight.sh.

Раньше список экспериментов был захардкожен прямо в shell-скрипте, а успех
шага определялся grep'ом по избранным строкам stdout (могло молча
пропустить реальную ошибку). Теперь эксперименты — данные (experiments/
*.yaml), а успех — реальный код возврата процесса (см. run_experiment.py).

Запуск:
    python3 scripts/run_batch.py                      # все experiments/*.yaml по порядку
    python3 scripts/run_batch.py --skip 2              # пропустить первые 2
    python3 scripts/run_batch.py --only 13_r18_soft_plateau
    python3 scripts/run_batch.py --experiments-dir experiments_custom/

Логи: logs/overnight_<timestamp>/ (как и раньше).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_experiment import run_experiment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments-dir", type=Path, default=ROOT / "experiments")
    p.add_argument("--skip", type=int, default=0, help="Пропустить первые N экспериментов")
    p.add_argument("--only", default=None, help="Запустить только эксперимент с этим name")
    return p.parse_args()


def main():
    args = parse_args()
    # Снижает фрагментацию CUDA-памяти между многочасовыми прогонами (RTX 3050 8GiB).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    exp_files = sorted(args.experiments_dir.glob("*.yaml"))
    if args.only:
        exp_files = [p for p in exp_files if p.stem == args.only]
    else:
        exp_files = exp_files[args.skip:]

    if not exp_files:
        sys.exit(f"Нет экспериментов для запуска в {args.experiments_dir} "
                 f"(skip={args.skip}, only={args.only})")

    log_dir = ROOT / "logs" / f"overnight_{time.strftime('%Y-%m-%d_%H-%M')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Логи: {log_dir}")
    print(f"Экспериментов: {len(exp_files)}")

    all_results = []
    for i, exp_path in enumerate(exp_files, 1):
        print(f"\n{'=' * 68}\n  [{i}/{len(exp_files)}] {exp_path.stem}\n{'=' * 68}")
        result = run_experiment(exp_path, log_dir)
        all_results.append(result)

    summary_path = log_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 68}\n  Все эксперименты завершены\n{'=' * 68}\n")
    print("Итоговые метрики:")
    for result in all_results:
        run_name = "run_" + result["name"]
        metrics_path = ROOT / "results" / run_name / "metrics_summary.json"
        if not metrics_path.exists():
            status = "OK" if result["train_ok"] else "ОШИБКА"
            print(f"  {run_name}: нет metrics_summary.json (train={status})")
            continue
        data = json.load(open(metrics_path, encoding="utf-8"))
        best = max(data, key=lambda x: x["val_f1"])
        print(f"  {run_name}: val_f1={best['val_f1']:.4f} (эп {best['epoch']})")

    print(f"\nЛоги: {log_dir}")
    n_ok = sum(1 for r in all_results if r["train_ok"] and r.get("visualize_ok") is not False)
    sys.exit(0 if n_ok == len(all_results) else 1)


if __name__ == "__main__":
    main()
