# CleanMARL

**CleanMARL** provides single-file, clean, and educational implementations of Deep Multi-Agent Reinforcement Learning (MARL) algorithms in PyTorch, following the same philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl).

### Main Features

* Implementations of key MARL algorithms: VDN, QMIX, COMA, MADDPG, FACMAC, IPPO, and MAPPO.

* A documentation for algorithms, code and training details.

* We support continuous and discrete actions.

* We support parallel environments and recurrent policies.

* Tensorboard and Weights & Biases logging.

We provide more details in our [documentation](https://cleanmarl-docs.readthedocs.io/en/latest/).

Check the `old_jax` branch for JAX implementations (non-jax envs only).

## Quick Start

Prerequisites:

* Python >=3.9

Installation:

```bash
git clone https://github.com/AmineAndam04/cleanmarl.git
cd cleanmarl
pip install .
```

To run experiment you can run for example:

```bash
python  cleanmarl/vdn.py --env_type="pz" --env_name="simple_spread_v3" --env_family="mpe" --use_wnb --wnb_project="cleanmarl-test" --wnb_entity="cleanmarl-test" --total_timesteps=1000000

python  cleanmarl/mappo.py --env_type="smaclite" --env_name="3m" 
```

## Supported Algorithms

| Algorithm | Variants Implemented |
|------------|----------------------|
| [Value Decomposition Networks (VDN)](https://arxiv.org/abs/1706.05296) | [`vdn.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/vdn.py) <br>  [`vdn_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/vdn_lstm.py) <br>  [`vdn_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/vdn_multienvs.py)|
| [QMIX](https://arxiv.org/abs/1803.11485) | [`qmix.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/qmix.py) <br> [`qmix_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/qmix_lstm.py) <br> [`qmix_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/qmix_multienvs.py)|
| [Counterfactual Multi-Agent (COMA)](https://arxiv.org/abs/1705.08926) | [`coma.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma.py)  <br> [`coma_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma_lstm.py) <br> [`coma_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma_multienvs.py) <br> [`coma_lstm_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma_lstm_multienvs.py)   |
| [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://arxiv.org/abs/1706.02275) | [`maddpg.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg.py) <br> [`maddpg_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg_multienvs.py) <br> [`maddpg_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg_lstm.py) <br> [`maddpg_continuous`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg_continuous.py) |
| [Factored Multi-Agent Centralized Policy Gradients (FACMAC)](https://arxiv.org/abs/2003.06709) | [`facmac.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/facmac.py) <br> [`facmac_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/facmac_multienvs.py) <br> [`facmac_continuous`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/facmac_continuous.py)|
| [Independent Proximal Policy Optimization (IPPO)](https://arxiv.org/abs/2011.09533) | [`ippo.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo.py) <br> [`ippo_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo_lstm.py) [`ippo_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo_multienvs.py) <br> [`ippo_lstm_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo_lstm_multienvs.py)   <br> [`ippo_continuous`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo_continuous.py)|
|  [Multi-Agent Proximal Policy Optimization (MAPPO)](https://arxiv.org/abs/2103.01955) | [`mappo.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo.py) <br> [`mappo_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo_lstm.py) <br> [`mappo_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo_multienvs.py) <br> [`mappo_lstm_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo_lstm_multienvs.py) <br> [`mappo_continuous`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo_continuous.py)|

## Supported environments

We use `marlbench` to interact with MARL environments. `marlbench` is a tool that provides (1) a common API for MARL envs, (2) vectorized envs, and (2) common wrappers (normalization, clipping ...)

Install using `uv pip install marlbench`

- Git repo: [marlbench](https://github.com/AmineAndam04/marlbench)

| Environment | Action space | Installation |
| ---  | --- | --- |
| [Level-Based Foraging](https://github.com/semitable/lb-foraging)  | Discrete | `pip install lbforaging` |
| [Multi-Robot Warehouse](https://github.com/semitable/robotic-warehouse)  | Discrete | `pip install rware` |
| [SMAClite](https://github.com/uoe-agents/smaclite) | Discrete | Install it from its GitHub repository |
| [PettingZoo](https://pettingzoo.farama.org/) | Discrete or continuous | `pip install pettingzoo` and install the extra dependencies for the family you use |
| [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/)  | Continuous | `pip install gymnasium-robotics` |
| [MAgent2](https://magent2.farama.org/)  | Discrete | `pip install magent2` |
| [SMAC](https://github.com/oxwhirl/smac) | Discrete | Follow the instructions in the SMAC repository |
| [SMACv2](https://github.com/oxwhirl/smacv2)  | Discrete | Follow the instructions in the SMACv2 repository |