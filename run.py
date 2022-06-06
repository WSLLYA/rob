from algorithm.ddpg import DDPG
from env.KukaGymEnv import KukaDiverseObjectEnv
import argparse
import sys, os
import pdb

import tensorflow as tf
import numpy as np

import random
from algorithm.My_toolkit import mkdir, wreplay, w_rate

from mpi4py import MPI

def comman_arg_parser():
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('--experiment_name', type=str, default='experiment', help="exp_name")
    parser.add_argument('--isrender', action="store_true",help="render GUI")
    parser.add_argument('--use_segmentation_Mask', help="evaluate model", action="store_true")
    parser.add_argument('--seed', type=int, default=0, help="random seed")

    # train
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size.default=64")
    parser.add_argument('--max_epochs', type=int, default=int(1e+4), help="max epochs of whole training")
    parser.add_argument('--noise_target_action', help="noise target action", action="store_true")
    parser.add_argument('--max_ep_steps', type=int, default=15, help="max steps of epoch")

    # priority
    parser.add_argument('--memory_size', type=int, default=1000, help="MEMORY_CAPACITY.default=1000")
    parser.add_argument("-p", "--priority", action="store_true", help="priority memory replay")
    parser.add_argument('--alpha', type=float, default=0.6, help="priority degree")
    parser.add_argument("--turn_beta",  action="store_true", help="turn the beta from 0.6 to 1.0")

    # evaluate
    parser.add_argument('--evaluation', action="store_true", help="Evaluation model")

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

def set_process_seeds(myseed):

    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)


def train(agent, env, eval_env, max_epochs, rank, nb_rollout_steps=15, inter_learn_steps=50, **kwargs):
    assert np.all(np.abs(env.action_space.low) == env.action_space.high)
    print('Pross_%d start rollout!'%(rank))
    with agent.sess.as_default(), agent.graph.as_default():
        obs = env.reset()
        # all objects's pos
        full_state = env._low_dim_full_state()
        episode_length = 0
        episodes = 0
        episode_cumulate_reward_history = []
        episode_cumulate_reward = 0
        eval_episodes = 0
        train_step = 0
        
        for epoch in range(int(max_epochs)):
            # rollouts
            for i in range(nb_rollout_steps):

                # predict next action
                action = agent.pi(obs)

                assert action.shape == env.action_space.shape

                new_obs, reward, done, info = env.step(action)

                new_full_state = env._low_dim_full_state()

                agent.num_timesteps += 1

                episode_cumulate_reward = 0.99 * episode_cumulate_reward + reward
                episode_length += 1

                agent.store_transition(obs, action, reward, new_obs, full_state,
                                        new_full_state, done, demo=False
                                        )
                obs = new_obs
                full_state = new_full_state

                # testing
                print("\nepoch_%d(%d): action:%s\nreward:%s"%(epoch,i,str(action),reward))

                if done:
                    #episode done
                    episodes += 1
                    agent.save_episode_result(episode_cumulate_reward, episode_length, info['is_success'], episodes)
                    episode_cumulate_reward_history.append(episode_cumulate_reward)
                    episode_cumulate_reward = 0
                    episode_length = 0
                    obs = env.reset()
                    break
            
            # train:replay learn
            if agent.pointer >= 5 * agent.batch_size:
                # print("\nReplay learning:", epoch)
                for t_train in range(inter_learn_steps):
                    agent.learn(train_step)
                    train_step += 1
                # agent.Save()

                # testing
                trans = agent.get_memory()
                path = os.getcwd()
                path = os.path.join(path, 'outs/replay_memory.txt')
                wreplay(trans[0], path)

            # evaluate
            sucess_epochs = 0
            sucess_rate = 0
            if eval_env is not None and epoch % 5 == 0:
                print("\n---------开始测试：--------")
                print('\neval_episodes:', eval_episodes)
                for eval_epoch in range(4):
                    
                    eval_obs, eval_done = eval_env.reset(), False

                    for i in range(nb_rollout_steps):
                        end_pos_before = env.get_ef_pos()
                        eval_action = agent.pi(eval_obs)
                        end_pos_after = env.get_ef_pos()
                        print("end_pos_before:", end_pos_before)
                        print("end_pos_after:", end_pos_after)
                        print("eval_action:", eval_action)
                        eval_obs, eval_reward, eval_done, eval_info = env.step(eval_action)
                        if eval_done:
                            break
                    if eval_info['is_success']:
                        print('Success!!!')
                        sucess_epochs += 1
                    else:
                        print('Fail!!!')
                sucess_rate = sucess_epochs / 50
                w_rate(eval_episodes, sucess_rate)
                eval_episodes += 1

            if (epoch + 1) % 100 == 0:
                agent.Save(epoch)

        return agent
    
def main(experiment_name, seed, max_epochs, evaluation, isrender, max_ep_steps, use_segmentation_Mask, **kwargs):
    
    #生成实验文件夹
    rank = MPI.COMM_WORLD.Get_rank()
    mkdir( rank, experiment_name)

    seed = seed + 2019 * rank
    set_process_seeds(seed)
    print('\nrank {}: seed={}'.format(rank, seed))

    env = KukaDiverseObjectEnv(renders=isrender,
                               isDiscrete=False,
                               maxSteps=max_ep_steps,
                               blockRandom=0.4,
                               removeHeightHack=True,
                               use_low_dim_obs=False,
                               use_segmentation_Mask=use_segmentation_Mask,
                               numObjects=4,
                               dv=1.0)
    kwargs['obs_space'] = env.observation_space
    kwargs['action_space'] = env.action_space
    kwargs['full_state_space'] = env.full_state_space

    if evaluation and rank == 0:
        eval_env = KukaDiverseObjectEnv(renders=False,
                                        isDiscrete=False,
                                        maxSteps=max_ep_steps,
                                        blockRandom=0.4,
                                        removeHeightHack=True,
                                        use_low_dim_obs=False,
                                        use_segmentation_Mask=use_segmentation_Mask,
                                        numObjects=4,
                                        dv=1.0)
    else:
        eval_env = None

    agent = DDPG(rank, **kwargs, experiment_name = experiment_name)

    agent_trained = train(agent, env, eval_env, max_epochs, rank, **kwargs)

    # if rank == 0:
    #     agent_trained.Save()

if __name__ == '__main__':
    args = comman_arg_parser()
    main(**args)
    


