import argparse
import gymnasium as gym
import torch
from src.config import BASE, DEVICE, DATA
from src.models.base import ACModel
from src.trainers.base_trainer import Trainer

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="MiniGrid-Unlock-v0")
    args = p.parse_args()
    env  = gym.make(args.env)
    model= ACModel(env.observation_space, env.action_space).to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=BASE['lr'])
    Trainer(args.env, model, opt,
            BASE['total_updates'], BASE['n_steps'], str(DATA/"base_stats.csv")).run()