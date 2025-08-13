# BlockRun Reinforcement Learning

This repository contains a minimal Deep Q-Network (DQN) example for the BlockRun environment.
The [`BlockRun_DQN.ipynb`](BlockRun_DQN.ipynb) notebook walks through training a DQN agent,
plotting its learning progress and lets you play the game yourself.

## Requirements
- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter Notebook](https://jupyter.org/)

Install the dependencies with:

```bash
pip install torch torchvision opencv-python matplotlib jupyter
```

## Getting started
1. Launch Jupyter Notebook in this directory:
   ```bash
   jupyter notebook
   ```
2. Open `BlockRun_DQN.ipynb` and run the cells sequentially to:
   - Initialise the environment.
   - Train a DQN agent.
   - Plot training rewards to visualise performance.
   - Play BlockRun manually (requires a windowing environment).

A pretrained model is saved to `q_network.pt` after training.

## Command line utilities
The scripts can also be used directly:

```bash
python dqn_training.py train       # train without visualisation
python dqn_training.py train_viz   # train with visualisation
python dqn_training.py play        # play manually
```
