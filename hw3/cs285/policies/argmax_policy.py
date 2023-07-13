import torch

from cs285.infrastructure import pytorch_util


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        # TODO return the action that maximizes the Q-value
        # at the current observation as the output
        observation = pytorch_util.from_numpy(observation)
        actions = self.critic.q_net(observation)
        action = torch.argmax(actions, dim=1)
        return action.squeeze()