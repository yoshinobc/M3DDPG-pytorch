from model import Policy, Critic
from memory import ReplayMemory
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import torch
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)

class Agent():
    def __init__(self, args, state_dim, action_space, name):
        self.args = args
        self.name = name
        self.actor = Policy(state_dim, action_space.n).to(args.device)
        self.critic = Critic(state_dim, action_space.n).to(args.device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.optimizer_actor = Adam(self.actor.parameters(), lr=args.actor_lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=args.critic_lr)

    def get_action(self, observation, greedy):
        action = self.actor(observation)
        if not greedy:
            action += torch.tensor(np.random.normal(0, 0.1),
                                   dtype=torch.float, device=self.args.device)
        return action

class M3DDPG():

    def __init__(self, args, env):
        self.args = args
        self.device = args.device
        self.obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        self.action_space = env.action_space[0]
        num_adversaries = min(env.n, args.num_adversaries)
        self.agents = []
        for i in range(num_adversaries):
            self.agents.append(Agent(args, self.obs_shape_n[i][0], env.action_space[0], f'good_{i}'))
        for i in range(num_adversaries, env.n):
            self.agents.append(Agent(args, self.obs_shape_n[i][0], env.action_space[0], f'bad_{i}'))
        self.memory = ReplayMemory(args.capacity)

    def add_memory_list(self, *args):
        transitions = Transition(*args)
        self.memory.append(transitions)

    def sample_action(self, state, greedy=False):
        actions = []
        for i, agent in enumerate(self.agents):
            observation_tensor = torch.tensor(
                state[i], dtype=torch.float, device=self.args.device).view(-1, self.obs_shape_n[i][0])
            action = agent.get_action(observation_tensor, greedy).squeeze(0).detach().cpu().numpy().tolist()
            actions.append(np.argmax(action))
        actions = np.array(actions)
        return actions

    def transition2batch(self, transitions):
        batch = Transition(*zip(*transitions))
        state_batch = torch.transpose(torch.tensor(
            batch.state, device=self.args.device, dtype=torch.float), 0, 1)
        actions = []
        for action in batch.action:
            action_vec = np.zeros(self.action_space.n)
            action_vec[np.argmax(action)] = 1
            actions.append(action_vec)
        action_batch = torch.tensor(
            actions, device=self.args.device, dtype=torch.float)
        next_state_batch = torch.transpose(torch.tensor(
            batch.next_state, device=self.args.device, dtype=torch.float), 0, 1)
        reward_batch = torch.tensor(
            batch.reward, device=self.args.device, dtype=torch.float)
        not_done = np.array([(not don) for don in batch.done])
        not_done_batch = torch.tensor(
            not_done, device=self.args.device, dtype=torch.float).unsqueeze(1)

        return state_batch, action_batch, next_state_batch, not_done_batch, reward_batch

    def update(self):
        actor_losses, critic_losses = [], []
        if self.memory.size() <= self.args.batch_size:
                return None, None
        transitions  = self.memory.sample(self.args.batch_size)
        state_n_batch, action_n_batch, next_state_n_batch, not_done_n_batch, reward_n_batch = self.transition2batch(transitions)
        for i, agent in enumerate(self.agents):
            if 'good' in agent.name:
                eps = self.args.eps
            else:
                eps = self.args.adv_eps

            reward_batch = reward_n_batch[i]
            not_done_batch = not_done_n_batch[i]


            #adv_critic_e:critic_e_targetを悪くするような摂動を通常のtarget出力に加える．
            #adv_actor_e:critic_eを悪くするような摂動を通常の出力に加える．
            #critic_pの更新に使うtargetQの入力に用いられる次行動

            _next_actions = [self.agents[j].actor(next_state_n_batch) for j in range(len(self.agents))]
            _next_action_n_batch_critic = torch.cat([_next_action if j != i else _next_action.detach() for j, _next_action in enumerate(_next_actions)],axis=1).squeeze(0)
            _critic_target_loss = self.agents[i].critic_target(next_state_n_batch, _next_action_n_batch_critic).mean()
            _critic_target_loss.backward()
            with torch.no_grad():
                next_action_n_batch_critic = torch.cat(
                    [_next_action + eps * _next_action.grad if j != i else _next_action for j, _next_action in enumerate(_next_actions)]
                    , axis=1).squeeze(0)

            #policy_pの更新に使うtargetQの入力に用いられる次行動

            _actions = [self.agents[j].actor(
                state_n_batch[j]) for j in range(len(self.agents))]

            #_action_n_batch_actor = [_action if j != i else _action.detach() for j, _action in enumerate(_actions)]
            _action_n_batch_actor = torch.cat([_action if j != i else _action.detach() for j, _action in enumerate(_actions)], axis=1)

            _actor_target_loss = self.agents[i].critic(
                state_n_batch, _action_n_batch_actor).mean()
            _actor_target_loss.backward()
            action_n_batch_actor = torch.cat(
                    [_action + eps * _action.grad if j != i else _action for j, _action in enumerate(_actions)], axis=1)


            #env update
            ##critic
            agent.optimizer_critic.zero_grad()
            currentQ = agent.critic(state_n_batch, action_n_batch)
            nextQ = agent.critic_target(next_state_n_batch, next_action_n_batch_critic)
            targetQ = reward_batch + self.args.gamma * not_done_batch * nextQ
            critic_loss = F.mse_loss(currentQ, targetQ)
            critic_loss.backward()
            agent.optimizer_critic.step()

            #print('b',agent.actor(state_n_batch[0]))
            ##policy
            agent.optimizer_actor.zero_grad()
            actor_loss = - agent.critic(state_n_batch, action_n_batch_actor).mean()
            actor_loss.backward()
            agent.optimizer_actor.step()
            #print('a',agent.actor(state_n_batch[0]))


            soft_update(agent.critic_target, agent.critic, self.args.tau)
            soft_update(agent.actor_target, agent.actor, self.args.tau)

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return actor_losses, critic_losses
