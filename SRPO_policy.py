import functools
import os

import d4rl
import gym
import numpy as np
import torch
import tqdm

import wandb
from dataset import D4RL_dataset
from SRPO import SRPO
from utils import get_args, marginal_prob_std, pallaral_simple_eval_policy


def train_policy(args, score_model, data_loader, start_epoch=0):
    n_epochs = args.n_policy_epochs
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 2
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = data_loader.sample(args.policy_batchsize)
            loss2 = score_model.update_SRPO_policy(data)
            avg_loss += 0.0
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            mean, std = pallaral_simple_eval_policy(score_model.SRPO_policy.select_actions,args.env,00)
            args.run.log({"eval/rew{}".format("deter"): mean}, step=epoch+1)
            args.run.log({"info/policy_q": score_model.SRPO_policy.q.detach().cpu().numpy()}, step=epoch+1)
            args.run.log({"info/lr": score_model.SRPO_policy_optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch+1)

def critic(args):
    for dir in ["./SRPO_policy_models"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./SRPO_policy_models", str(args.expid))):
        os.makedirs(os.path.join("./SRPO_policy_models", str(args.expid)))
    run = wandb.init(project="SRPO_policy", name=str(args.expid))
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

    # TODO
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
    critic(args)