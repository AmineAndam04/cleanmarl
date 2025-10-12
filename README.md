# CleanMARL

**CleanMARL** provides single-file, clean, and educational implementations of Deep Multi-Agent Reinforcement Learning (MARL) algorithms in PyTorch, following the same philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl).

### Main Features:
* Implementations of key MARL algorithms: VDN, QMIX, COMA, MADDPG, FACMAC, IPPO, and MAPPO.

* A documentation for algorithms, code and training details.

* We support parallel environments and recurrent policies.

* Tensorboard and Weights & Biases logging.

We provide more details in our [documentation](https://cleanmarl-docs.readthedocs.io/en/latest/).

You can find runs in Weights & Biases: [Weights & Biases](https://api.wandb.ai/links/andamamine-cleanmarl/b6a74ca6)
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

## Algorithms Implemented

| Algorithm | Variants Implemented |
|------------|----------------------|
| [Value Decomposition Networks (VDN)](https://arxiv.org/abs/1706.05296) | [`vdn.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/vdn.py) <br>  [`vdn_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/vdn_lstm.py) <br>  [`vdn_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/vdn_multienvs.py)|
| [QMIX](https://arxiv.org/abs/1803.11485) | [`qmix.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/qmix.py) <br> [`qmix_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/qmix_lstm.py) <br> [`qmix_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/qmix_multienvs.py) <br>  [`qmix_memefficient.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/qmix_memefficient.py) |
| [Counterfactual Multi-Agent (COMA)](https://arxiv.org/abs/1705.08926) | [`coma.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma.py)  <br> [`coma_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma_lstm.py) <br> [`coma_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma_multienvs.py) <br> [`coma_lstm_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma_lstm_multienvs.py) <br>  [`coma_lbf.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/coma_lbf.py) |
| [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://arxiv.org/abs/1706.02275) | [`maddpg.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg.py) <br> [`maddpg_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg_multienvs.py) <br> [`maddpg_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg_lstm.py) <br> [`maddpg_lstm_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/maddpg_lstm_multienvs.py) |
| [Factored Multi-Agent Centralized Policy Gradients (FACMAC)](https://arxiv.org/abs/2003.06709) | [`facmac.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/facmac.py) <br> [`facmac_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/facmac_multienvs.py)|
| [Independent Proximal Policy Optimization (IPPO)](https://arxiv.org/abs/2011.09533) | [`ippo.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo.py) <br> [`ippo_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo_lstm.py) <br> [`ippo_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo_multienvs.py) <br> [`ippo_lstm_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/ippo_lstm_multienvs.py) |
|  [Multi-Agent Proximal Policy Optimization (MAPPO)](https://arxiv.org/abs/2103.01955) | [`mappo.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo.py) <br> [`mappo_lstm.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo_lstm.py) <br> [`mappo_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo_multienvs.py) <br> [`mappo_lstm_multienvs.py`](https://github.com/AmineAndam04/cleanmarl/blob/main/cleanmarl/mappo_lstm_multienvs.py) |

