import argparse
import os
from mpi4py import MPI
import tensorflow as tf
from env.KukaGymEnv import KukaDiverseObjectEnv
from algorithm.ddpg import DDPG
import numpy as np


def common_arg_parser():
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('--memory_size', type=int, default=1000, help="MEMORY_CAPACITY.default=1000")
    parser.add_argument('--experiment_name', type=str, default='experiment', help="exp_name")
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size.default=16")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--max_epochs', type=int, default=int(1e+4), help="max epochs of whole training")
    parser.add_argument('--noise_target_action', help="noise target action", action="store_true")
    parser.add_argument('--max_ep_steps', type=int, default=20, help="max steps of epoch")
    parser.add_argument('--evaluation', action="store_true", help="evaluate model")
    parser.add_argument('--isrender', action="store_true",help="render GUI")
    parser.add_argument('--use_segmentation_Mask', help="evaluate model", action="store_true")

    # priority
    parser.add_argument("-p", "--priority", action="store_true", help="priority memory replay")
    parser.add_argument('--alpha', type=float, default=0.6, help="priority degree")
    parser.add_argument("--turn_beta",  action="store_true", help="turn the beta from 0.6 to 1.0")

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


def set_process_seeds(myseed):

    tf.set_random_seed(myseed)
    np.random.seed(myseed)

def show_test(seed, experiment_name, max_ep_steps, use_segmentation_Mask, **kwargs):

    rank = MPI.COMM_WORLD.Get_rank()
    seed = seed + 2019 * rank
    set_process_seeds(seed)
    env = KukaDiverseObjectEnv(renders=True,
                               isDiscrete=False,
                               maxSteps=max_ep_steps,
                               blockRandom=0.4,
                               removeHeightHack=True,
                               use_low_dim_obs=False,
                               use_segmentation_Mask=use_segmentation_Mask,
                               numObjects=1,
                               dv=1.0)
    kwargs['obs_space'] = env.observation_space
    kwargs['action_space'] = env.action_space
    kwargs['full_state_space'] = env.full_state_space

    agent = DDPG(rank, **kwargs, experiment_name = experiment_name)

    agent.load()
    with agent.sess.as_default(), agent.graph.as_default():
        os.system("clear")
        print('-------------------------------------------')
        print("experiment_name:", experiment_name)
        print('-------------------------------------------')

        episode_num = 0
        success_num = 0
        obs, done = env.reset(), False

        for _ in range(3000):
            action = agent.pi(obs)
            print("\n")
            print("action:", action)

            # testing
            env.get_ob_pos()

            obs, reward, done, info = env.step(action)
            if done:
                episode_num +=1
                obs = env.reset()
                if info['is_success']:
                    success_num +=1
        print("%d success in %d episodes. success rate %f"%(success_num, episode_num, success_num/episode_num))
    print("ok!")


if __name__ == '__main__':
    args = common_arg_parser()
    show_test(**args)

