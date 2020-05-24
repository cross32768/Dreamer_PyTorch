import argparse
import json
import os
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import Agent
from model import Encoder, RecurrentStateSpaceModel, ActionModel
from wrappers import GymWrapper, RepeatAction


def main():
    parser = argparse.ArgumentParser(description='Test learned model')
    parser.add_argument('dir', type=str, help='log directory to load learned model')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('-R', '--action-repeat', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()

    # define environment and apply wrapper
    env = suite.load(args.domain_name, args.task_name)
    env = pixels.Wrapper(env, render_kwargs={'height': 64,
                                             'width': 64,
                                             'camera_id': 0})
    env = GymWrapper(env)
    env = RepeatAction(env, skip=args.action_repeat)

    # define models
    with open(os.path.join(args.dir, 'args.json'), 'r') as f:
        train_args = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(train_args['state_dim'],
                                    env.action_space.shape[0],
                                    train_args['rnn_hidden_dim']).to(device)
    action_model = ActionModel(train_args['state_dim'], train_args['rnn_hidden_dim'],
                               env.action_space.shape[0]).to(device)

    # load learned parameters
    encoder.load_state_dict(torch.load(os.path.join(args.dir, 'encoder.pth')))
    rssm.load_state_dict(torch.load(os.path.join(args.dir, 'rssm.pth')))
    action_model.load_state_dict(torch.load(os.path.join(args.dir, 'action_model.pth')))

    # define agent
    policy = Agent(encoder, rssm, action_model)

    # test learnged model in the environment
    for episode in range(args.episodes):
        policy.reset()
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if args.render:
                env.render(height=256, width=256, camera_id=0)

        print('Total test reward at episode [%4d/%4d] is %f' %
              (episode+1, args.episodes, total_reward))


if __name__ == '__main__':
    main()
