from cs285.policies.MLP_policy import MLPPolicy
from numpy.lib.twodim_base import triu_indices
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super().__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation)
        dist = self(observation)
        if sample:
            action = ptu.to_numpy(dist.sample().detach())
            return np.clip(action, self.action_range[0], self.action_range[1])
        return ptu.to_numpy(dist.mean.detach())

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing log probs and apply correction for Tanh squashing

        # HINT:
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file
        mean = self.mean_net(observation)
        # logstd = torch.tanh(self.logstd)
        clipped_log = np.clip(self.logstd.data.cpu().numpy(), self.log_std_bounds[0], self.log_std_bounds[1])
        self.logstd.data = ptu.from_numpy(clipped_log).to(ptu.device)
        dist = sac_utils.SquashedNormal(mean, torch.exp(self.logstd))
        return dist

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        # update alpha
        obs = ptu.from_numpy(obs)

        self.log_alpha_optimizer.zero_grad()
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = torch.mean(dist.log_prob(action), dim=-1, keepdim=True)
        # log_prob = dist.log_prob(action)
        alpha_loss = torch.mean(-1 * self.alpha * (log_prob + self.target_entropy))
        # print("log_prob", log_prob.shape)
        # print("alpha", self.alpha.shape)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.optimizer.zero_grad()
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = torch.mean(dist.log_prob(action), dim=-1, keepdim=True)
        # log_prob = dist.log_prob(action)
        q1, q2 = critic(obs, action)
        actor_loss = torch.mean(self.alpha * log_prob - torch.min(q1, q2))
        # print("log_prob", log_prob.shape)
        # print("q1", q1.shape)
        actor_loss.backward()
        self.optimizer.step()

        # return actor_loss.detach().cpu(), alpha_loss.detach().cpu(), self.alpha.detach().cpu()
        return ptu.to_numpy(actor_loss), ptu.to_numpy(alpha_loss), ptu.to_numpy(self.alpha)


