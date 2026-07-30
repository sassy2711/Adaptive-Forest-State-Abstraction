# Adaptive based state aggregation for Reinforcement Learning

This repository contains an implementation of the paper "Adaptive state aggregation for reinforcement learning by K. -S. Hwang, Y. -J. Chen and W. -C. Jiang", enhanced with a novel modification that incorporates the use of random forests.

## How to run

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. To compare the existing model with the novel model:
    ```bash
    python train_test.py --profile= [cartpole/acrobot]
    ```
3. To use the models you can import from `adaptive_agents.py`

4. To modify hyperparameters or experiment settings, edit the `config.yaml` file.

