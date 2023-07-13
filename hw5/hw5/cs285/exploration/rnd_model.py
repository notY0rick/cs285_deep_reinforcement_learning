from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
import numpy as np
from torch import nn
import torch


def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()


def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_1)
        self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_2)
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()

    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        # torch.tensor
        if isinstance(ob_no, np.ndarray):
            ob_no = ptu.from_numpy(ob_no)
        f_out = self.f(ob_no).detach()
        fh_out = self.f_hat(ob_no)
        assert f_out.shape == fh_out.shape, "f and f_hat shape not equal!"
        # return torch.norm(f_out - fh_out, dim=1)
        return torch.sqrt(torch.mean((f_out - fh_out)**2, dim=1))
        # return torch.mean((f_out - fh_out)**2, dim=1)

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        self.optimizer.zero_grad()
        ob_no = ptu.from_numpy(ob_no)
        err = self(ob_no)
        loss = self.loss(err, torch.zeros_like(err))
        # loss = torch.mean(err * err, dim=0)
        # loss = torch.mean(err)
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        return loss
