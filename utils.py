import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

def build_net(layer_shape, hidden_activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
		super(Actor, self).__init__()
		layers = [state_dim] + list(hid_shape)

		self.a_net = build_net(layers, hidden_activation, output_activation)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def forward(self, state, deterministic, with_logprob):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
		# we learn log_std rather than std, so that exp(log_std) is always > 0
		std = torch.exp(log_std)
		dist = Normal(mu, std)
		if deterministic: u = mu
		else: u = dist.rsample()

		'''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
		a = torch.tanh(u)
		if with_logprob:
			# Get probability density of logp_pi_a from probability density of u:
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a

class Double_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2

#reward engineering for better training
def Reward_adapter(r):
	# For Pendulum-v0
	return r


def Action_adapter(a,max_action):
	#from [-1,1] to [-max,max]
	return  a*max_action

def Action_adapter_reverse(act,max_action):
	#from [-max,max] to [-1,1]
	return  act/max_action


def evaluate_policy(env, max_action, agent, turns, steps, run_id):
	total_scores = 0
	# draw_flag = False
	for j in range(turns):
		s,vo,wo,pt,pos= env.reset()
		done = False
		count = 0

		# if not draw_flag:
		Action_array = np.zeros((int(env.Time / env.actionTime), 3))
		V_start_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
		E_start_array = np.zeros((int(env.Time / env.actionTime)+1, 3))
		W_start_array = np.zeros((int(env.Time / env.actionTime)+1, 3))
		P_start_array = np.zeros((int(env.Time / env.actionTime)+1, 3))
		V_body_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
		W_body_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
		PT_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
		Dis_array = np.zeros((int(env.Time/env.actionTime)+1, 3))
	
		s,vo,wo,pt,pos= env.reset()
		V_start_array[count, :] = vo
		W_start_array[count, :] = wo
		P_start_array[count, :] = pos
		V_body_array[count,:] = s[0:3]
		W_body_array[count,:] = s[3:6]
		E_start_array[count, :] = s[6:9]
		Dis_array[count,:] = s[9:12]
		PT_array[count,:] = pt
		count+=1
	
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			act = Action_adapter(a, max_action)
			s_next, r, done, vo_, wo_, pt_, fr, fl, rb0, pos_ = env.step(act)

			# if not draw_flag:
			Action_array[count-1, :] = [fr,fl,rb0]
			V_start_array[count, :] = vo_
			W_start_array[count, :] = wo_
			P_start_array[count, :] = pos_
			V_body_array[count,:] = s_next[0:3]
			W_body_array[count,:] = s_next[3:6]
			E_start_array[count, :] = s_next[6:9]
			Dis_array[count,:] = s_next[9:12]
			pos_ = np.array([pos_[0],0,pos_[2]])
			PT_array[count,:] = pt_
			count += 1
			total_scores += r
			s = s_next
		# if not draw_flag:
		time = np.arange(0, env.Time + 2*env.actionTime, env.actionTime)
		os.makedirs(f'pic/path{run_id}', exist_ok=True)
		os.makedirs(f'pic/all_plots{run_id}', exist_ok=True)
		path_path = f'./pic/path{run_id}/path_{steps}_{j}.png'
		env.path(path_path,P_start_array,PT_array)
		path = f'./pic/all_plots{run_id}/all_plots_{steps}_{j}.png'
		env.pathall(time,V_start_array,W_start_array,E_start_array,P_start_array,Action_array,V_body_array,W_body_array,Dis_array,path)
		# draw_flag = True
		
	return int(total_scores/turns)


def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	
