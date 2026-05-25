# StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems

## Overview

StepFinder is a lightweight framework for step-level failure attribution in LLM-based multi-agent systems (MAS). It encodes execution logs into temporal semantic sequences and applies a parameter-efficient combination of temporal modeling and attention modules to identify the root cause step.

## Requirements

Python 3.10 or higher is required. To install requirements:

```bash
pip install -r requirements.txt
```

## Data Preparation

The training trajectories are provided in this repository under `data/`. For evaluation, download the Who&When benchmark test sets from [their repository](https://github.com/mingyin1/Agents_Failure_Attribution) and place them as follows:

```
data/
├── Algorithm-Generated/
│   ├── train/      # provided
│   └── test/       # from Who&When
└── Hand-Crafted/
    ├── train/      # provided
    └── test/       # from Who&When
```

We construct training trajectories using a trajectory regeneration strategy: given an initial failure trajectory, an LLM regenerates a different reasoning path and root cause error while preserving the original question, ground truth, and agent team. The prompt template is provided in `prompts.py` (`TRAJECTORY_REGENERATION_PROMPT`), which can also be used to generate custom training data.

## Step 1: Feature Construction

Encode execution logs into temporal semantic sequences using Qwen3-Embedding-0.6B. Encoded features are cached automatically and reused in subsequent runs.

```bash
# Algorithm-Generated
python feature_construction.py --data_dir ./data/Algorithm-Generated/train
python feature_construction.py --data_dir ./data/Algorithm-Generated/test

# Hand-Crafted
python feature_construction.py --data_dir ./data/Hand-Crafted/train
python feature_construction.py --data_dir ./data/Hand-Crafted/test
```

## Step 2: Training and Evaluation

StepFinder is trained and evaluated on each subset independently. The best checkpoint is saved automatically based on accuracy.

```bash
# Algorithm-Generated
python main.py \
    --train_dir ./data/Algorithm-Generated/train \
    --test_dir ./data/Algorithm-Generated/test \
    --save_path model/Algorithm-Generated_best_model.pth

# Hand-Crafted
python main.py \
    --train_dir ./data/Hand-Crafted/train \
    --test_dir ./data/Hand-Crafted/test \
    --save_path model/Hand-Crafted_best_model.pth
```

The key hyperparameters are listed below.

| Hyperparameter      | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `--alpha`           | Controls the strength of agent-aware bias and gating ($\alpha$ in the paper) |
| `--beta`            | Weight of the multi-scale difference term ($\beta$ in the paper) |
| `--scales`          | Temporal scales for multi-scale differencing ($s$ in the paper) |
| `--gamma`           | Weight of the position bias term ($\gamma$ in the paper)     |
| `--lambda_temporal` | Weight of the temporal consistency loss ($\lambda$ in the paper) |

## LLM-based Baseline

To prompt an LLM to produce a ranked list of candidate root cause steps,  we provide the prompt template in `prompts.py` (`ALL_AT_ONCE_RANKING_PROMPT`).

## Citation

```bibtex
@inproceedings{zhu2026stepfinder,
  title={StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems},
  author={Taiyu Zhu and Yifan Wu and Weilin Jin and Ying Li and Gang Huang},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  year={2026}
}
```