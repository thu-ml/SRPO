import os
import gym
import d4rl
import scipy
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

from utils import get_args, pallaral_simple_eval_policy, marginal_prob_std
from dataset import D4RL_dataset

from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import simple_eval_policy

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



class MapPolicy(nn.Module):
    def __init__(self, action_dim, state_dim, layer=2):
        super().__init__()
        self.net = mlp([state_dim] + [256]*layer + [action_dim], output_activation=nn.Tanh)

    def forward(self, state):
        return self.net(state)
    def select_actions(self, state):
        return self(state)
    
class GaussPolicy(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        self.net = mlp([state_dim, 256, 256, action_dim], output_activation=nn.Tanh)
        self.sigma_param = nn.Parameter(torch.zeros(action_dim,)) # (action_dim, 1)

    def forward(self, state):
        mu = self.net(state)
        shape = [1] * len(state.shape)
        shape[1] = -1
        sigma = (torch.clamp(self.sigma_param.view(shape) + torch.zeros_like(mu), -5.0,100.0)).exp()
        return mu, sigma
    def select_actions(self, state):
        return self(state)[0]
    

class GaussV(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = mlp([state_dim, 256, 256, 1], output_activation=nn.Tanh)
        self.net_sigma = mlp([state_dim, 256, 256, 1], output_activation=None)

    def forward(self, state):
        mu = self.net(state)
        sigma = (torch.clamp(self.net_sigma(state), -5.0,10.0)).exp()
        self.sigma = sigma
        return mu


class DIQL(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        self.diffusion_policy = MapPolicy(output_dim, input_dim-output_dim, layer=args.policy_layer).to("cuda")
        self.diffusion_policy_optimizer = torch.optim.Adam(self.diffusion_policy.parameters(), lr=3e-4)
        self.diffusion_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    def update(self, data):
        s = data['s']

        self.diffusion_behavior.eval()
        a = self.diffusion_policy(s)
        if self.args.t > 1.0:
            t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        elif self.args.t > 0.0:
            t = torch.ones(a.shape[0], device=s.device) * self.args.t
        else:
            assert False
        
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()
            if "noise" in args.WT:
                episilon = episilon - z

        # TODO rewrite
        if "VDS" in args.WT:
            wt = std ** 2
        elif "stable" in args.WT:
            wt = 1.0
        elif "score" in args.WT:
            wt = alpha_t / std
        elif "hight" in args.WT:
            wt = std ** 4
        else:
            assert False

        detach_a = a.detach().requires_grad_(True)
        qs = self.q[0].q0_target.both(detach_a , s)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.diffusion_policy.q = torch.mean(q)
        # TODO be aware that there is a small std gap term here, this seem won't affect final performance though
        # guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach() * std[..., None]
        guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        # self.q_guidance_norm = torch.sum(guidance ** 2, dim=-1).mean()
        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        if "2term" in self.args.WT:
            loss = ((episilon - guidance* self.args.alpha) * a).sum(-1) * wt
        else:
            loss = (episilon * a).sum(-1) * wt - (guidance * a).sum(-1) * self.args.alpha

        loss = loss.mean()
        self.diffusion_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.diffusion_policy_optimizer.step()
        self.diffusion_policy_lr_scheduler.step()
        self.diffusion_behavior.train()

class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)




class Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        assert sdim > 0
        if args.q_layer == 2:
            self.q0 = TwinQ2(adim, sdim).to(args.device)
        elif args.q_layer == 3:
            self.q0 = TwinQ(adim, sdim).to(args.device)
        elif args.q_layer == 4:
            self.q0 = TwinQ4(adim, sdim).to(args.device)
        else:
            assert False
        print(args.q_layer)
        self.q0_target = copy.deepcopy(self.q0).to(args.device)

        self.vf = ValueFunction(sdim).to("cuda")
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)
        self.discount = 0.99
        self.args = args
        self.alpha = args.alpha
        if args.tau is None:
            self.tau = 0.9 if "maze" in args.env else 0.7
        else:
            assert False
        print(self.tau)

    def update_q0(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        # Update Q function
        targets = r + (1. - d.float()) * self.discount * next_v.detach()
        if "ablate_action_noise" in args.expid:
            qs = self.q0.both(a + (torch.randn_like(a) * 0.2).clamp(-0.5, 0.5), s)
        else:
            qs = self.q0.both(a, s)
        self.v = v.mean()
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        
        # Update target
        update_target(self.q0, self.q0_target, 0.005)

def train_policy(args, score_model, data_loader, start_epoch=0):
    n_epochs = args.n_policy_epochs
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 2

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = data_loader.sample(args.policy_batchsize)
            loss2 = score_model.update(data)
            avg_loss += 0.0
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            mean, std = pallaral_simple_eval_policy(score_model.diffusion_policy.select_actions,args.env,00)
            args.run.log({"eval/rew{}".format("deter"): mean}, step=epoch+1)
            args.run.log({"info/policy_q": score_model.diffusion_policy.q.detach().cpu().numpy()}, step=epoch+1)
            args.run.log({"info/lr": score_model.diffusion_policy_optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch+1)

def critic(args):
    for dir in ["./DIQL_policy_models"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./DIQL_policy_models", str(args.expid))):
        os.makedirs(os.path.join("./DIQL_policy_models", str(args.expid)))
    run = wandb.init(project="DIQL_policy", name=str(args.expid))
    wandb.config.update(args)
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    args.run = run
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=args.beta)
    args.marginal_prob_std_fn = marginal_prob_std_fn

    score_model= DIQL(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)

    args.actor_load_path = "path/to/yout/ckpt/file"
    if args.actor_load_path is not None:
        print("loading actor...")
        ckpt = torch.load(args.actor_load_path, map_location=args.device)
        score_model.load_state_dict({k:v for k,v in ckpt.items() if "diffusion_behavior" in k}, strict=False)

    args.critic_load_path = "path/to/yout/ckpt/file"
    if args.critic_load_path is not None:
        print("loadind critic...")
        ckpt = torch.load(args.critic_load_path, map_location=args.device)
        score_model.q[0].load_state_dict(ckpt)

    dataset = D4RL_dataset(args)

    print("training critic")
    train_policy(args, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()

if __name__ == "__main__":
    args = get_args()
    if "0.02" in args.WT:
        args.t = 0.02
    elif "0.98" in args.WT:
        args.t = 0.98
    critic(args)