from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from SAC import SAC_countinuous
from env_stright_xoz_frfl import RGBlimpenv
import shutil
import argparse
import torch
import numpy as np
import os



'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, default='16', help='run id')
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    # Build Env
    Time = 100.0
    action_time = 0.2
    target = np.array([40.0, 0.0, 20.0])
    env = RGBlimpenv(Time, action_time, target)
    eval_env = RGBlimpenv(Time, action_time, target)

    opt.state_dim = env.state_dim
    opt.action_dim = env.action_dim
    opt.max_action = env.max_action   #remark: action space【-max,max】
    opt.max_e_steps = int(Time/action_time)


    # Seed Everything
    torch.manual_seed(opt.seed)
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = '_' + timenow[0:13]  + timenow[-2::]
        writepath = f'runs/{opt.run_id}{timenow}'
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary

    # if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    total_steps = 0
    while total_steps < opt.Max_train_steps:
        s, *_ = env.reset()
        done = False

        '''Interact & trian'''
        while not done:
            a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
            act = Action_adapter(a, opt.max_action)  # act∈[-max,max]
            s_next, r, done, *_ = env.step(act)

            r = Reward_adapter(r)

            agent.replay_buffer.add(s, a, r, s_next, done)
            s = s_next
            total_steps += 1

            '''train if it's time'''
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                for j in range(opt.update_every):
                    agent.train()

            '''record & log'''
            if total_steps % opt.eval_interval == 0:

                ep_r = evaluate_policy(eval_env, opt.max_action, agent, turns=3, steps = total_steps, run_id = opt.run_id)
                if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')

            '''save model'''
            if total_steps % opt.save_interval == 0:
                agent.save('Blimp', int(total_steps/1000))

if __name__ == '__main__':
    main()
