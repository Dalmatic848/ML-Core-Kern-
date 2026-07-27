#!/usr/bin/env python3
"""
Запускает один эксперимент из experiments/<name>.yaml: обучение (train.py) +
визуализация (visualize_results.py/visualize_soft.py) — с проверкой
РЕАЛЬНОГО кода возврата процесса, а не grep по строкам stdout, как раньше
в run_overnight.sh (там успех/ошибка определялись совпадением текстовых
паттернов, что маскировало часть реальных сбоев).

Запуск:
    python3 scripts/run_experiment.py experiments/13_r18_soft_plateau.yaml
    python3 scripts/run_experiment.py experiments/13_r18_soft_plateau.yaml --log-dir logs/overnight_2026-07-27_10-00
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def run_step(cmd: list, log_path: Path) -> tuple:
    """(success, duration_s). Полный stdout+stderr пишется в log_path без
    фильтрации — grep по избранным строкам (как раньше) прячет часть ошибок."""
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT)
    return proc.returncode == 0, time.time() - start


def run_experiment(exp_path: Path, log_dir: Path) -> dict:
    with open(exp_path, encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    name = exp["name"]
    python = sys.executable
    result = {"name": name, "train_ok": False, "visualize_ok": None}

    print(f"→ [{name}] train...")
    train_cmd = [python, "train.py"] + [str(a) for a in exp["train_args"]]
    train_log = log_dir / f"{name}_train.log"
    ok, dur = run_step(train_cmd, train_log)
    result["train_ok"] = ok
    result["train_duration_s"] = round(dur, 1)
    if not ok:
        print(f"  ✗ train ОШИБКА — см. {train_log}")
        return result
    print(f"  ✓ train завершён ({dur:.0f}s)")

    vis_script = exp.get("visualize_script")
    if vis_script:
        print(f"→ [{name}] visualize...")
        vis_cmd = [python, vis_script] + [str(a) for a in exp.get("visualize_args", [])]
        vis_log = log_dir / f"{name}_vis.log"
        ok2, dur2 = run_step(vis_cmd, vis_log)
        result["visualize_ok"] = ok2
        result["visualize_duration_s"] = round(dur2, 1)
        if ok2:
            print(f"  ✓ visualize завершена ({dur2:.0f}s)")
        else:
            print(f"  ✗ visualize ОШИБКА — см. {vis_log}")

    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", type=Path, help="Путь к experiments/<name>.yaml")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="По умолчанию: logs/adhoc_<timestamp>/")
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir or (ROOT / "logs" / f"adhoc_{time.strftime('%Y-%m-%d_%H-%M')}")
    log_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(args.experiment, log_dir)
    with open(log_dir / f"{result['name']}_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    ok = result["train_ok"] and result.get("visualize_ok") is not False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
