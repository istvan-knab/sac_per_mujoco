import torch
import torch.optim as optim
import torch.nn.functional as F

from replay_memory.replay_memory import ReplayMemory
from neural_networks.actor_nertwork import Actor
from neural_networks.critic_network import Critic
from train_setup.seed_all import seed_all, test_seed
class SoftActorCritic:
    def __init__(self, config, env):
        self.config = config
        self.set_memory()
        self.tau = config["TAU"]
        self.actor = Actor(env, config, hidden_dim=config["HIDDEN_LAYERS"])
        self.critic_1 = Critic(env, config, hidden_dim=config["HIDDEN_LAYERS"])
        self.critic_2 = Critic(env, config, hidden_dim=config["HIDDEN_LAYERS"])
        self.critic_1_target = Critic(env, config, hidden_dim=config["HIDDEN_LAYERS"])
        self.critic_2_target = Critic(env, config, hidden_dim=config["HIDDEN_LAYERS"])
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["LR"])
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=config["LR"])
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=config["LR"])
        seed_all(self.config["SEED"], env)

    def set_memory(self):

        if self.config["TRAIN_MODE"] == "simple":
            self.memory = ReplayMemory(self.config["BUFFER_SIZE"], self.config["BATCH_SIZE"])
        elif self.config["TRAIN_MODE"] == "per":
            raise NotImplementedError
        elif self.config["TRAIN_MODE"] == "ucb":
            raise NotImplementedError
        else:
            raise ValueError(f"Choose train mode from:\n -simple \n -per \n -ucb")


    def update_policy(self):

        if self.memory.__len__() < self.config["BATCH_SIZE"]:
            return 0, 0, 0

        state, action, next_state, reward, done = self.memory.sample()
        state = state.to(self.config["DEVICE"])
        action = action.to(self.config["DEVICE"])
        next_state = next_state.to(self.config["DEVICE"])
        reward = reward.to(self.config["DEVICE"])
        done = done.to(self.config["DEVICE"])
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample_action(next_state)
            next_log_prob = torch.clamp(next_log_prob, min=1e-10).float()
            q1_target = self.critic_1_target(next_state, next_action)
            q2_target = self.critic_2_target(next_state, next_action)
            q_target = reward + ~done * self.config["DISCOUNT_FACTOR"] * (
                        torch.min(q1_target, q2_target) - self.config["ENTROPY_COEFFICIENT"] * next_log_prob)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        critic_1_loss = F.mse_loss(q1, q_target.float())
        critic_2_loss = F.mse_loss(q2, q_target.float())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_2_optimizer.step()

        # Update Actor
        new_action, log_prob = self.actor(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        log_prob = torch.clamp(log_prob, min=1e-10).float()
        actor_loss = (self.config["ENTROPY_COEFFICIENT"] * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_1_loss, critic_2_loss, actor_loss