import functools
import os

import d4rl
import gym
import numpy as np
import torch
import tqdm

import wandb
from dataset import D4RL_dataset
from SRPO import SRPO_Behavior
from utils import get_args, marginal_prob_std


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
            loss2 = score_model.update_behavior(data)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.state_dict(), os.path.join("./SRPO_model_factory", str(args.expid), "behavior_ckpt{}.pth".format(epoch+1)))

def behavior(args):
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
    score_model= SRPO_Behavior(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

    dataset = D4RL_dataset(args)

    print("training behavior")
    train_behavior(args, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()

if __name__ == "__main__":
    args = get_args()
    behavior(args)
