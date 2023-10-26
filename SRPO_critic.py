import os

import d4rl
import gym
import numpy as np
import torch
import tqdm

import wandb
from dataset import D4RL_dataset
from SRPO import SRPO_IQL
from utils import get_args, pallaral_simple_eval_policy


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

    score_model= SRPO_IQL(input_dim=state_dim+action_dim, output_dim=action_dim, args=args).to(args.device)
    score_model.q[0].to(args.device)

    dataset = D4RL_dataset(args)

    print("training critic")
    train_critic(args, score_model, dataset, start_epoch=0)
    print("finished")
    run.finish()

if __name__ == "__main__":
    args = get_args()
    critic(args)