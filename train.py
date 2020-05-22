import argparse
from datetime import datetime
import json
import os
from pprint import pprint
import time
import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import Agent
from model import (Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel,
                   ValueModel, ActionModel)
from utils import ReplayBuffer, preprocess_obs, lambda_return
from wrappers import GymWrapper, RepeatAction


def main():
    parser = argparse.ArgumentParser(description='Dreamer for DM control')
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--test-interval', type=int, default=10)
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=2)
    parser.add_argument('--state-dim', type=int, default=30)
    parser.add_argument('--rnn-hidden-dim', type=int, default=200)
    parser.add_argument('--buffer-capacity', type=int, default=1000000)
    parser.add_argument('--all-episodes', type=int, default=1000)
    parser.add_argument('-S', '--seed-episodes', type=int, default=5)
    parser.add_argument('-C', '--collect-interval', type=int, default=100)
    parser.add_argument('-B', '--batch-size', type=int, default=50)
    parser.add_argument('-L', '--chunk-length', type=int, default=50)
    parser.add_argument('-H', '--imagination-horizon', type=int, default=15)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--model_lr', type=float, default=6e-4)
    parser.add_argument('--value_lr', type=float, default=8e-5)
    parser.add_argument('--action_lr', type=float, default=8e-5)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--clip-grad-norm', type=int, default=100)
    parser.add_argument('--free-nats', type=int, default=3)
    parser.add_argument('--action-noise-var', type=float, default=0.3)
    args = parser.parse_args()

    # Prepare logging
    log_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M'))
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    pprint(vars(args))
    writer = SummaryWriter(log_dir=log_dir)

    # define env and apply wrappers
    env = suite.load(args.domain_name, args.task_name)
    env = pixels.Wrapper(env, render_kwargs={'height': 64,
                                             'width': 64,
                                             'camera_id': 0})
    env = GymWrapper(env)
    env = RepeatAction(env, skip=args.action_repeat)

    # define replay buffer
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                 observation_shape=env.observation_space.shape,
                                 action_dim=env.action_space.shape[0])

    # define models and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(args.state_dim,
                                    env.action_space.shape[0],
                                    args.rnn_hidden_dim).to(device)
    obs_model = ObservationModel(args.state_dim, args.rnn_hidden_dim).to(device)
    reward_model = RewardModel(args.state_dim, args.rnn_hidden_dim).to(device)
    model_params = (list(encoder.parameters()) +
                    list(rssm.parameters()) +
                    list(obs_model.parameters()) +
                    list(reward_model.parameters()))
    model_optimizer = Adam(model_params, lr=args.model_lr, eps=args.eps)

    # define value model and action model and optimizer
    value_model = ValueModel(args.state_dim, args.rnn_hidden_dim).to(device)
    action_model = ActionModel(args.state_dim, args.rnn_hidden_dim,
                               env.action_space.shape[0]).to(device)
    value_optimizer = Adam(value_model.parameters(),
                           lr=args.value_lr, eps=args.eps)
    action_optimizer = Adam(action_model.parameters(),
                            lr=args.action_lr, eps=args.eps)

    # collect seed episodes with random action
    for episode in range(args.seed_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs    

    # main training loop
    for episode in range(args.seed_episodes, args.all_episodes):
        # collect experiences
        start = time.time()
        policy = Agent(encoder, rssm, action_model)

        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(obs)
            action += np.random.normal(0, np.sqrt(args.action_noise_var),
                                        env.action_space.shape[0])
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
            total_reward += reward

        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, args.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.time() - start))

        # update model parameters and value model and action model parameters
        start = time.time()
        for update_step in range(args.collect_interval):
            observations, actions, rewards, _ = \
                replay_buffer.sample(args.batch_size, args.chunk_length)

            # preprocess observations and transpose tensor for RNN training
            observations = preprocess_obs(observations)
            observations = torch.as_tensor(observations, device=device)
            observations = observations.transpose(3, 4).transpose(2, 3)
            observations = observations.transpose(0, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)

            # embed observations with CNN
            embedded_observations = encoder(
                observations.reshape(-1, 3, 64, 64)).view(args.chunk_length, args.batch_size, -1)

            # prepare Tensor to maintain states sequence and rnn hidden states sequence
            states = torch.zeros(
                args.chunk_length, args.batch_size, args.state_dim, device=device)
            rnn_hiddens = torch.zeros(
                args.chunk_length, args.batch_size, args.rnn_hidden_dim, device=device)

            # initialize state and rnn hidden state with 0 vector
            state = torch.zeros(args.batch_size, args.state_dim, device=device)
            rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)

            # compute state and rnn hidden sequences and kl loss
            kl_loss = 0
            for l in range(args.chunk_length-1):
                next_state_prior, next_state_posterior, rnn_hidden = \
                    rssm(state, actions[l], rnn_hidden, embedded_observations[l+1])
                state = next_state_posterior.rsample()
                states[l+1] = state
                rnn_hiddens[l+1] = rnn_hidden
                kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
                kl_loss += kl.clamp(min=args.free_nats).mean()
            kl_loss /= (args.chunk_length - 1)

            # states[0] and rnn_hiddens[0] are always 0 and have no information
            states = states[1:]
            rnn_hiddens = rnn_hiddens[1:]

            # compute reconstructed observations and predicted rewards
            flatten_states = states.view(-1, args.state_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, args.rnn_hidden_dim)
            recon_observations = obs_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length-1, args.batch_size, 3, 64, 64)
            predicted_rewards = reward_model(flatten_states, flatten_rnn_hiddens).view(
                args.chunk_length-1, args.batch_size, 1)

            # compute loss for observation and reward
            obs_loss = 0.5 * mse_loss(
                recon_observations, observations[1:], reduction='none').mean([0, 1]).sum()
            reward_loss = 0.5 * mse_loss(predicted_rewards, rewards[:-1])

            # add all losses and update model parameters with gradient descent
            model_loss = kl_loss + obs_loss + reward_loss
            model_optimizer.zero_grad()
            model_loss.backward()
            clip_grad_norm_(model_params, args.clip_grad_norm)
            model_optimizer.step()

            # compute target values
            flatten_states = flatten_states.detach()
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()
            imaginated_states = torch.zeros(args.imagination_horizon + 1,
                                            *flatten_states.shape,
                                            device=flatten_states.device)
            imaginated_rnn_hiddens = torch.zeros(args.imagination_horizon + 1,
                                                 *flatten_rnn_hiddens.shape,
                                                 device=flatten_rnn_hiddens.device)
            imaginated_states[0] = flatten_states
            imaginated_rnn_hiddens[0] = flatten_rnn_hiddens

            for h in range(1, args.imagination_horizon + 1):
                actions = action_model(flatten_states, flatten_rnn_hiddens)
                flatten_states_prior, flatten_rnn_hiddens = rssm.prior(flatten_states,
                                                                       actions,
                                                                       flatten_rnn_hiddens)
                flatten_states = flatten_states_prior.rsample()
                imaginated_states[h] = flatten_states
                imaginated_rnn_hiddens[h] = flatten_rnn_hiddens

            flatten_imaginated_states = imaginated_states.view(-1, args.state_dim)
            flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(-1, args.rnn_hidden_dim)
            imaginated_rewards = \
                reward_model(flatten_imaginated_states,
                             flatten_imaginated_rnn_hiddens).view(args.imagination_horizon + 1, -1)
            imaginated_values = \
                value_model(flatten_imaginated_states,
                            flatten_imaginated_rnn_hiddens).view(args.imagination_horizon + 1, -1)
            lambda_value_target = lambda_return(imaginated_rewards, imaginated_values,
                                                args.gamma, args.lambda_)
        
            # update_value model
            value_loss = 0.5 * mse_loss(imaginated_values, lambda_value_target.detach())
            value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            clip_grad_norm_(value_model.parameters(), args.clip_grad_norm)
            value_optimizer.step()

            # update value model and action model
            action_loss = -1 * (lambda_value_target.mean())
            action_optimizer.zero_grad()
            action_loss.backward()
            clip_grad_norm_(action_model.parameters(), args.clip_grad_norm)
            action_optimizer.step()

            # print losses and add tensorboard
            print('update_step: %3d model loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: % .5f, action_loss: %.5f value_loss: %.5f'
                  % (update_step+1,
                     model_loss.item(), kl_loss.item(), obs_loss.item(), reward_loss.item(), action_loss.item(), value_loss.item()))
            total_update_step = episode * args.collect_interval + update_step
            writer.add_scalar('model loss', model_loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)
            writer.add_scalar('action loss', action_loss.item(), total_update_step)
            writer.add_scalar('value loss', value_loss.item(), total_update_step)

        print('elasped time for update: %.2fs' % (time.time() - start))

        # test to get score without exploration noise
        if (episode + 1) % args.test_interval == 0:
            policy = Agent(encoder, rssm, action_model)
            start = time.time()
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy(obs, training=False)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            writer.add_scalar('total reward at test', total_reward, episode)
            print('Total test reward at episode [%4d/%4d] is %f' %
                  (episode+1, args.all_episodes, total_reward))
            print('elasped time for test: %.2fs' % (time.time() - start))

    # save learned model parameters
    torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pth'))
    torch.save(rssm.state_dict(), os.path.join(log_dir, 'rssm.pth'))
    torch.save(obs_model.state_dict(), os.path.join(log_dir, 'obs_model.pth'))
    torch.save(reward_model.state_dict(), os.path.join(log_dir, 'reward_model.pth'))
    torch.save(action_model.state_dict(), os.path.join(log_dir, 'action_model.pth'))
    torch.save(value_model.state_dict(), os.path.join(log_dir, 'value_model.pth'))
    writer.close()

if __name__ == '__main__':
    main()
