import gymnasium as gym
import yaml
from adaptive_agents import AdaptiveForestAgent, AdaptiveStateAggregator  # Adjust as needed

import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(inf1, inf2, interval=10, title="Mean Reward vs. Number of Episodes"):
    
    episodes = np.arange(interval, interval * len(inf1) + 1, interval)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, inf1, label='Tree Agent', marker='o')
    plt.plot(episodes, inf2, label='Forest Agent', marker='x')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Mean Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_results.png")



import argparse
parser = argparse.ArgumentParser(description="Choose the environment profile")
parser.add_argument("--profile", type=str, default="acrobot", choices=["cartpole", "acrobot"], help="Environment profile to use")
args = parser.parse_args()
profile = args.profile

# Load YAML config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

profile_cfg = config["env_profiles"][profile]
env_name = profile_cfg["env"]

env = gym.make(env_name)
env.reset()

common_params = profile_cfg["common"]
training_episodes = profile_cfg["training"]["episodes"]

forest_agent = AdaptiveForestAgent(
    n_trees=profile_cfg["forest_agent"]["n_trees"],
    feature_fraction=profile_cfg["forest_agent"]["feature_fraction"],
    state_dim=env.observation_space.shape[0],
    actions=list(range(env.action_space.n)),
    **common_params
)

tree_agent = AdaptiveStateAggregator(
    state_dim=env.observation_space.shape[0],
    actions=list(range(env.action_space.n)),
    **common_params
)

rewards, inf1 = tree_agent.train(env, episodes=training_episodes)
inf2 = forest_agent.train(env, episodes=training_episodes)

plot_training_results(inf1, inf2, interval=10, title=f"Mean Reward vs. Number of Episodes ({profile.capitalize()})")
