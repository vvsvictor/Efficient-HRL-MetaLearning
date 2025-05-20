import copy, random, torch
import gymnasium as gym
from torch.distributions import Categorical
from src.models.feudal import FuNModel
from src.config import DEVICE, MAML

class MetaTrainer:
    def __init__(self, tasks, vocab_size=64, meta_batch=3, inner_steps=1):
        self.tasks      = tasks
        env0           = gym.make(tasks[0])
        self.model     = FuNModel(env0.observation_space, env0.action_space)
        self.mis_embed = torch.nn.Embedding(vocab_size, 32)
        self.opt       = torch.optim.Adam(
            list(self.model.parameters()) + list(self.mis_embed.parameters()),
            lr=MAML['meta_lr'])
        self.inner_steps = inner_steps
        self.vocab_size = vocab_size
        self._mission2id = {}

    def _mission_id(self, mission):
        if mission not in self._mission2id:
            self._mission2id[mission] = len(self._mission2id) % self.vocab_size
        return self._mission2id[mission]

    def _adapt(self, fast):
        opt = torch.optim.SGD(
            list(fast.parameters()) + list(self.mis_embed.parameters()),
            lr=MAML['inner_lr'])
        # rollouts + compute_losses as in feudal
        return

    def _meta_loss(self, fast):
        # rollouts + compute_losses on query set
        return torch.tensor(0.0)

    def train_iteration(self):
        losses = []
        for env_id in random.sample(self.tasks, self.inner_steps):
            fast = copy.deepcopy(self.model)
            self._adapt(fast)
            losses.append(self._meta_loss(fast))
        meta_loss = torch.stack(losses).mean()
        self.opt.zero_grad(); meta_loss.backward(); self.opt.step()
        return meta_loss.item()