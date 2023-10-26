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
import copy

from utils import get_args, pallaral_simple_eval_policy, marginal_prob_std
from dataset import D4RL_dataset

from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F





class SRPO(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        self.deter_policy = MapPolicy(output_dim, input_dim-output_dim).to("cuda")
        self.deter_policy_optimizer = torch.optim.Adam(self.deter_policy.parameters(), lr=3e-4)
        self.deter_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.deter_policy_optimizer, T_max=1500000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    def update_iql(self, data):
        all_a = data['a']
        all_s = data['s']
        s = all_s[:256]
        a = all_a[:256]
        self.q[0].update_q0(data)
        with torch.no_grad():
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
        adv = target_q - v
        temp = 10.0 if "maze" in self.args.env else 3.0
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0) # should be 90-150

        policy_out = self.deter_policy(s)
        bc_losses = torch.sum((policy_out - a)**2, dim=1)
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        self.deter_policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.deter_policy_optimizer.step()
        self.deter_policy_lr_scheduler.step()
        self.policy_loss = policy_loss


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
        self.tau = 0.9 if "maze" in args.env else 0.7
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

def train_critic(args, score_model, data_loader, start_epoch=0):
    n_epochs = 150
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    # evaluation_inerval = 4
    evaluation_inerval = 1
    save_interval = 10

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = data_loader.sample(256)
            loss2 = score_model.update_iql(data)
            avg_loss += 0.0
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            if (epoch % 5 == 4) or epoch==0:
                mean, std = pallaral_simple_eval_policy(score_model.deter_policy.select_actions,args.env,00)
                args.run.log({"eval/rew{}".format("deter"): mean}, step=epoch+1)
            args.run.log({"loss/v_loss": score_model.q[0].v_loss.detach().cpu().numpy()}, step=epoch+1)
            args.run.log({"loss/q_loss": score_model.q[0].q_loss.detach().cpu().numpy()}, step=epoch+1)
            args.run.log({"loss/q": score_model.q[0].q.detach().cpu().numpy()}, step=epoch+1)
            args.run.log({"loss/v": score_model.q[0].v.detach().cpu().numpy()}, step=epoch+1)
            args.run.log({"loss/policy_loss": score_model.policy_loss.detach().cpu().numpy()}, step=epoch+1)
            args.run.log({"info/lr": score_model.deter_policy_optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch+1)
        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_model_factory", str(args.expid), "critic_ckpt{}.pth".format(epoch+1)))

def critic(args):
    for dir in ["./SRPO_model_factory"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./SRPO_model_factory", str(args.expid))):
        os.makedirs(os.path.join("./SRPO_model_factory", str(args.expid)))
    run = wandb.init(project="SRPO_model_factory", name=str(args.expid))
    wandb.config.update(args)
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.run = run
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn

    score_model= SRPO(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)

    dataset = D4RL_dataset(args)

    print("training critic")
    train_critic(args, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()

if __name__ == "__main__":
    args = get_args()
    critic(args)