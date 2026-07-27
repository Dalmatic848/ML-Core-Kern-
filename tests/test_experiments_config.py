"""experiments/*.yaml — эксперимент как данные (заменяет хардкод, который
раньше жил в run_overnight.sh) и scripts/run_experiment.py — реальный код
возврата процесса вместо grep по stdout."""

from pathlib import Path

import yaml

from scripts.run_experiment import run_step

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"


def test_at_least_one_experiment_defined():
    files = list(EXPERIMENTS_DIR.glob("*.yaml"))
    assert len(files) > 0


def test_every_experiment_yaml_has_required_keys():
    for path in EXPERIMENTS_DIR.glob("*.yaml"):
        with open(path, encoding="utf-8") as f:
            exp = yaml.safe_load(f)
        assert "name" in exp, path
        assert "train_args" in exp, path
        assert isinstance(exp["train_args"], list), path
        assert exp["name"] == path.stem, f"{path}: name={exp['name']!r} != имя файла"
        if "visualize_script" in exp:
            assert (ROOT / exp["visualize_script"]).exists(), (
                f"{path}: visualize_script={exp['visualize_script']!r} не существует"
            )


def test_run_step_reports_success(tmp_path):
    log_path = tmp_path / "ok.log"
    ok, duration = run_step(["python3", "-c", "print('hi')"], log_path)
    assert ok is True
    assert duration >= 0
    assert "hi" in log_path.read_text(encoding="utf-8")


def test_run_step_reports_failure_via_real_exit_code(tmp_path):
    """Раньше run_overnight.sh определял ошибку через grep текстовых
    паттернов — если сообщение не совпадало ни с одним паттерном, сбой
    молча пропускался. run_step проверяет реальный returncode."""
    log_path = tmp_path / "fail.log"
    ok, _ = run_step(["python3", "-c", "import sys; sys.exit(1)"], log_path)
    assert ok is False
