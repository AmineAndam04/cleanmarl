import json
import subprocess
import sys


def test_ippo(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            "--save_model",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
    run_dirs = list(tmp_path.glob("IPPO-*"))
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


def test_ippo_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            "--device=cuda",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_lbf(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo.py",
            "--env_type=lbf",
            "--env_name=Foraging-2s-10x10-4p-2f-v3",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_pz(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo.py",
            "--env_type=pz",
            "--env_name=simple_spread_v3",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_lstm(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--tbptt=3",
            "--eval-steps=100",
            "--save_model",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
    run_dirs = list(tmp_path.glob("IPPO-*"))
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


def test_ippo_lstm_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            "--tbptt=3",
            "--device=cuda",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_lstm_lbf(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm.py",
            "--env_type=lbf",
            "--env_name=Foraging-2s-10x10-4p-2f-v3",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--tbptt=3",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_lstm_pz(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm.py",
            "--env_type=pz",
            "--env_name=simple_spread_v3",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--tbptt=3",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_multienv(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_multienvs.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            "--save_model",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
    run_dirs = list(tmp_path.glob("IPPO-*"))
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


def test_ippo_multienv_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_multienvs.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            "--device=cuda",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_multienv_lbf(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_multienvs.py",
            "--env_type=lbf",
            "--env_name=Foraging-2s-10x10-4p-2f-v3",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_lstm_multienvs(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm_multienvs.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--tbptt=3",
            "--eval-steps=100",
            "--save_model",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
    run_dirs = list(tmp_path.glob("IPPO-*"))
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


def test_ippo_lstm_multienvs_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm_multienvs.py",
            "--env_type=smaclite",
            "--env_name=3m",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--eval-steps=100",
            "--tbptt=3",
            "--device=cuda",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_lstm_multienvs_lbf(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm_multienvs.py",
            "--env_type=lbf",
            "--env_name=Foraging-2s-10x10-4p-2f-v3",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--tbptt=3",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )


def test_ippo_lstm_multienvs_pz(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "cleanmarl/ippo_lstm_multienvs.py",
            "--env_type=pz",
            "--env_name=simple_spread_v3",
            "--total-timesteps=1000",
            "--n_episodes=2",
            "--tbptt=3",
            "--eval-steps=100",
            f"--work_dir={tmp_path}",
        ],
        check=True,
    )
