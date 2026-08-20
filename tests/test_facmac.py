import json
import subprocess
import sys

import torch


def test_facmac(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/facmac.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            "--save_model",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
    run_dirs = list(tmp_path.glob("FACMAC-*"))
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    checkpoint_path = run_dir / "agent.pt"
    args_path = run_dir / "args.json"

    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 0
    assert args_path.is_file()

    # Verify the saved configuration.
    saved_args = json.loads(args_path.read_text())
    assert saved_args["save_model"] is True
    assert saved_args["env_type"] == "smaclite"
    assert saved_args["env_name"] == "3m"
    assert saved_args["total_timesteps"] == 1000


def test_facmac_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/facmac.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            "--device=cuda",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_facmac_lbf(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/facmac.py",
            "--env_type=lbf",
            "--env_name=Foraging-2s-10x10-4p-2f-v3",
            "--total-timesteps=1000",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_facmac_pz(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/facmac.py",
            "--env_type=pz",
            "--env_name=simple_spread_v3",
            "--total-timesteps=1000",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_facmac_multienv(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/facmac_multienvs.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            "--save_model",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
    run_dirs = list(tmp_path.glob("FACMAC-*"))
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    checkpoint_path = run_dir / "agent.pt"
    args_path = run_dir / "args.json"

    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 0
    assert args_path.is_file()
    # Verify the saved configuration.
    saved_args = json.loads(args_path.read_text())
    assert saved_args["save_model"] is True
    assert saved_args["env_type"] == "smaclite"
    assert saved_args["env_name"] == "3m"
    assert saved_args["total_timesteps"] == 1000


def test_facmac_multienv_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/facmac_multienvs.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            "--device=cuda",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_facmac_multienv_lbf(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/facmac_multienvs.py",
            "--env_type=lbf",
            "--env_name=Foraging-2s-10x10-4p-2f-v3",
            "--total-timesteps=1000",
            "--buffer-size=100",
            "--batch-size=4",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
