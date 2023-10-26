import os
import gym
import d4rl
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

from utils import get_args, marginal_prob_std
from dataset import D4RL_dataset

from model import *
import torch
import torch.nn as nn

class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else None

    def forward(self, x, training=False):
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x

class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, out_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=256, activations=F.relu):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim+128, self.hidden_dim)

        self.blocks = nn.ModuleList([MLPResNetBlock(self.hidden_dim, self.activations, self.dropout_rate, self.use_layer_norm)
                                     for _ in range(self.num_blocks)])

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, training=False):
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.activations(x)
        x = self.out_fc(x)

        return x
    
    
class ScoreNet_IDQL(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim))
        self.device=args.device
        self.marginal_prob_std = marginal_prob_std
        self.args=args
        self.main = MLPResNet(args.actor_blocks, input_dim, output_dim, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, activations=nn.Mish())
        self.cond_model = mlp([64, 128, 128], output_activation=None, activation=nn.Mish)

        # The swish activation function
        # self.act = lambda x: x * torch.sigmoid(x)
        
    def forward(self, x, t, condition):
        embed = self.cond_model(self.embed(t))
        all = torch.cat([x, condition, embed], dim=-1)
        h = self.main(all)
        return h

class DIQL(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, args=None):
        super().__init__()
        self.alphas = []
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
    
    def update(self, data):
        self.step += 1
        all_a = data['a']
        all_s = data['s']

        # Update diffusion behavior
        self.diffusion_behavior.train()


        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - 1e-3) + 1e-3  
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        episilon = self.diffusion_behavior(perturbed_x, random_t, all_s)
        loss = torch.mean(torch.sum((episilon - z)**2, dim=(1,)))
        self.loss =loss

        self.diffusion_optimizer.zero_grad()
        loss.backward()  
        self.diffusion_optimizer.step()




def train_behavior(args, score_model, data_loader, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    # evaluation_inerval = 4
    evaluation_inerval = 1
    save_interval = 20

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = data_loader.sample(2048)
            loss2 = score_model.update(data)
            avg_loss += 0.0
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.state_dict(), os.path.join("./DIQL_model_factory", str(args.expid), "behavior_ckpt{}.pth".format(epoch+1)))

def behavior(args):
    for dir in ["./DIQL_model_factory"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./DIQL_model_factory", str(args.expid))):
        os.makedirs(os.path.join("./DIQL_model_factory", str(args.expid)))
    run = wandb.init(project="DIQL_model_factory", name=str(args.expid))
    wandb.config.update(args)
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.run = run
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=args.beta)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    # initial a new IQL class
    score_model= DIQL(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

    assert args.actor_blocks == 3

    dataset = D4RL_dataset(args)

    print("training behavior")
    train_behavior(args, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()

if __name__ == "__main__":
    args = get_args()
    behavior(args)