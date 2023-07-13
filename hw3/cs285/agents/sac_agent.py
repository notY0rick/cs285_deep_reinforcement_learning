from collections import OrderedDict

from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure import sac_utils
import cs285.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super().__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the 
        next_ob_no = ptu.from_numpy(next_ob_no)
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        self.critic.optimizer.zero_grad()
        dist = self.actor(next_ob_no)
        next_ac_na = dist.rsample()
        next_ac_na_prob = torch.mean(dist.log_prob(next_ac_na), dim=1, keepdim=True).squeeze()
        tq1, tq2 = self.critic_target(next_ob_no, next_ac_na)
        tq = torch.min(torch.concat((tq1, tq2), dim=1), dim=1)[0] # torch.min(tq1, tq2)
        target = re_n + self.gamma * (tq - self.actor.alpha * next_ac_na_prob) * (1 - terminal_n)
        target = target.detach()
        q1, q2 = self.critic(ob_no, ac_na)
        # print("next_ac_na_prob", next_ac_na_prob.shape)
        # print("tq", tq.shape)
        # print("re_n", re_n.shape)
        # print("target", target.shape)
        # print("q1", q1.shape)
        loss1 = self.critic.loss(q1.squeeze(), target)
        loss2 = self.critic.loss(q2.squeeze(), target)
        loss = loss1 + loss2
        loss.backward()
        self.critic.optimizer.step()

        return ptu.to_numpy(loss.cpu() / 2.)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging

        critic_loss = []
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss.append(self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n))
        
        if self.training_step % self.critic_target_update_frequency == 0:
            sac_utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        actor_loss = []
        alpha_loss = []
        alpha = []
        if self.training_step % self.actor_update_frequency == 0:
            for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
                ac_loss, al_loss, al = self.actor.update(ob_no, self.critic)
                actor_loss.append(ac_loss)
                alpha_loss.append(al_loss)
                alpha.append(al)

        loss = OrderedDict()
        loss['Critic_Loss'] = np.mean(critic_loss)
        loss['Actor_Loss'] = np.mean(actor_loss)
        loss['Alpha_Loss'] = np.mean(alpha_loss)
        loss['Temperature'] = np.mean(alpha)
        self.training_step += 1
        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
