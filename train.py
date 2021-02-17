import argparse
import gym
from m3ddpg import M3DDPG
import numpy as np
import pickle
import os
import json
import torch
import datetime
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import time

def make_env(args):
    scenario = scenarios.load(f'{args.env_name}.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation)
    return env

def make_action(actions, action_space):
    agent_actions = []
    for i, action in enumerate(actions):
        action_vec = np.zeros(action_space[i].n)
        action_vec[action] = 1
        agent_actions.append(action_vec)
    return agent_actions

def set_seed(seed, env):
    torch.manual_seed(seed)
    env.seed = seed
    np.random.seed(seed)
    return env

def evaluate(m3ddpg, args):
    env = make_env(args)
    env = set_seed(args.seed+100, env)
    total_reward = [0] * env.n
    for _ in range(args.evaluate_num):
        state_n = env.reset()
        for _ in range(args.max_episode_len):
            action_n = m3ddpg.sample_action(state_n, greedy=True)
            agent_actions = make_action(action_n, env.action_space)
            next_state_n, reward_n, done_n, _ = env.step(agent_actions)
            if args.render:
                env.render()
            time.sleep(0.1)
            for i in range(env.n):
                total_reward[i] += reward_n[i]
            if all(done_n):
                state_n = env.reset()
                break
    if args.render:
        env.close()
    return np.mean(np.array(total_reward)).tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='simple')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--capacity', default='1e6', type=float)
    parser.add_argument('--steps', type=float, default=1e6)
    parser.add_argument('--max_episode_len', type=int, default=25)
    parser.add_argument('--start_steps', type=float, default=1e3)
    parser.add_argument('--evaluate-interval', type=float, default=1e3)
    parser.add_argument('--evaluate_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--adv-eps', default=1e-3, type=float)
    parser.add_argument('--num-adversaries', type=int, default=0)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=5e-3, type=float)
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--memo', default='', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('---render', action='store_true')

    args = parser.parse_args()
    dt_now = datetime.datetime.now()
    args.logger_dir = f'experiments/{args.env_name}_{args.seed}_{dt_now}'
    os.makedirs(args.logger_dir, exist_ok=False)
    with open('{}/hyperparameters.json'.format(args.logger_dir), 'w') as f:
        f.write(json.dumps(args.__dict__))

    env = make_env(args)
    env = set_seed(args.seed, env)
    m3ddpg = M3DDPG(args, env)

    total_reward_list = []
    reward_mean_list = []

    step = 0
    episode_step = 0

    state_n = env.reset()
    total_reward = [0] * env.n
    actor_loss_list = []
    critic_loss_list = []
    while True:
        if step <= args.start_steps:
            action_n = np.array([env.action_space[i].sample() for i in range(env.n)])
        else:
            action_n = m3ddpg.sample_action(state_n)
        agent_action = make_action(action_n, env.action_space)
        next_state_n, reward_n, done_n, _ = env.step(agent_action)
        for i in range(env.n):
            total_reward[i] += reward_n[i]
        m3ddpg.add_memory_list(state_n, action_n, next_state_n, reward_n, done_n)
        episode_step += 1
        if step >= args.start_steps:
            actor_loss, critic_loss = m3ddpg.update()
            if actor_loss is not None:
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

        state_n = next_state_n

        if step % args.evaluate_interval == 0:
            reward_mean= evaluate(m3ddpg, args)
            print('====================')
            print(f'step: {step}  reward: {reward_mean}')
            print('====================')
            reward_mean_list.append(reward_mean)
            results = {
                'train_reward': total_reward_list,
                'reward_mean_list': reward_mean_list,
                }
            pickle.dump(results, open(
                '{}/results.pkl'.format(args.logger_dir), 'wb'))

        if step > args.steps:
            break
        step += 1
        if all(done_n) or episode_step > args.max_episode_len:
            total_reward_list.append(total_reward)
            actor_loss = np.mean(actor_loss_list, axis=0)
            critic_loss = np.mean(critic_loss_list, axis=0)
            print(f'step: {step}  reward: {total_reward}  actor loss: {actor_loss}  critic loss: {critic_loss}')
            total_reward = [0] * env.n
            episode_step = 0
            state_n = env.reset()
            actor_loss_list = []
            critic_loss_list = []

    results = {
        'train_reward': total_reward_list,
        'reward_mean_list': reward_mean_list,
    }

    pickle.dump(results, open(
                                '{}/results{}.pkl'.format('results', args.seed), 'wb'))

    pickle.dump(results, open(
        '{}/results.pkl'.format(args.logger_dir), 'wb'))

if __name__ == '__main__':
    main()
