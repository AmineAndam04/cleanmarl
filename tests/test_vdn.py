import subprocess
import sys


def test_vdn(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/vdn.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--learning-starts=10",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_vdn_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/vdn.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--learning-starts=10",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            "--device=cuda",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_vdn_lbf(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/vdn.py",
            "--env_type=lbf",
            "--env_name=Foraging-2s-10x10-4p-2f-v3",
            "--total-timesteps=1000",
            "--learning-starts=10",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_vdn_pz(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/vdn.py",
            "--env_type=pz",
            "--env_name=simple_spread_v3",
            "--total-timesteps=1000",
            "--learning-starts=10",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
