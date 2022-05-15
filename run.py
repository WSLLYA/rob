from algorithm.ddpg import DDPG
from env.KukaGymEnv import KukaDiverseObjectEnv
import argparse

import tensorflow as tf
import numpy as np

import random
from algorithm.My_toolkit import mkdir
from algorithm.My_toolkit import wreplay

from mpi4py import MPI

def comman_arg_parser():
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument('--memory_size', type=int, default=1000, help="MEMORY_CAPACITY.default=1000")
    parser.add_argument('--experiment_name', type=str, default='experiment', help="exp_name")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size.default=64")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--max_epochs', type=int, default=int(1e+4), help="max epochs of whole training")
    parser.add_argument('--noise_target_action', help="noise target action", action="store_true")
    parser.add_argument('--max_ep_steps', type=int, default=20, help="max steps of epoch")
    parser.add_argument('--isrender', action="store_true",help="render GUI")
    parser.add_argument('--use_segmentation_Mask', help="evaluate model", action="store_true")
    parser.add_argument('--evaluation', action="store_true", help="Evaluation model")

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
    random.seed(myseed)

def train(agent, env, max_epochs, rank, nb_rollout_steps=15, inter_learn_steps=10, **kwargs):
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
            for i in range(nb_rollout_steps):

                # predict next action
                action = agent.pi(obs)
                
                # testing
                print(action)

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
                print("\nepoch【%d】【%d】: action:%s,reward:%s"%(epoch,i,str(action),reward))

                if done:
                    #episode done
                    episodes += 1
                    agent.save_episode_result(episode_cumulate_reward, episode_length, info['is_success'], episodes)
                    episode_cumulate_reward_history.append(episode_cumulate_reward)
                    episode_cumulate_reward = 0
                    episode_length = 0
                    obs = env.reset()
                    break
            
            # replay learn
            if agent.pointer >= 5 * 10:
                print("\nReplay learning:", epoch)
                for t_train in range(inter_learn_steps):
                    agent.learn(train_step)
                    train_step += 1
                agent.Save()

                # testing
                trans = agent.get_memory()
                wreplay(trans[0])
                    

            # evaluate
            # if eval_env is not None and epoch % 5 == 0:
            #     print('\neval_episodes:', eval_episodes)
            #     eval_episode_cumulate_reward = 0
            #     eval_episode_length = 0
                
            #     eval_obs, eval_done = eval_env.reset(), False
            #     while not eval_done:
            #         eval_action = agent.pi(eval_obs)
            #         print("eval_action:", eval_action)
            #         eval_obs, eval_reward, eval_done, eval_info = env.step(eval_action)
            #         eval_episode_cumulate_reward = 0.99 * eval_episode_cumulate_reward + eval_reward
            #         eval_episode_length += 1
            #     eval_episodes += 1
            #     agent.save_eval_episode_result(eval_episode_cumulate_reward, eval_episode_length,
            #                                    eval_info['is_success'], eval_episodes)
            #     if eval_info['is_success']:
            #         print('Success!!!')
            #     else:
            #         print('Fail!!!')

        return agent
    
def main(experiment_name, seed, max_epochs, evaluation, isrender, max_ep_steps, use_segmentation_Mask, **kwargs):
    
    #生成实验文件夹
    rank = MPI.COMM_WORLD.Get_rank()
    mkdir( rank, experiment_name)

    seed = seed + 2019 * rank
    set_process_seeds(seed)
    print('\nrank {}: seed={}'.format(rank, seed))

    env = KukaDiverseObjectEnv(renders=True,
                               isDiscrete=False,
                               maxSteps=max_ep_steps,
                               blockRandom=0.4,
                               removeHeightHack=True,
                               use_low_dim_obs=False,
                               use_segmentation_Mask=use_segmentation_Mask,
                               numObjects=5,
                               dv=1.0)
    kwargs['obs_space'] = env.observation_space
    kwargs['action_space'] = env.action_space
    kwargs['full_state_space'] = env.full_state_space

    # if evaluation and rank == 0:
    #     eval_env = KukaDiverseObjectEnv(renders=isrender,
    #                                     isDiscrete=False,
    #                                     maxSteps=max_ep_steps,
    #                                     blockRandom=0.4,
    #                                     removeHeightHack=True,
    #                                     use_low_dim_obs=False,
    #                                     use_segmentation_Mask=use_segmentation_Mask,
    #                                     numObjects=1,
    #                                     dv=1.0)
    # else:
    #     eval_env = None
    agent = DDPG(rank, **kwargs, experiment_name = experiment_name)

    agent_trained = train(agent, env, max_epochs, rank, **kwargs)

if __name__ == '__main__':
    args = comman_arg_parser()
    main(**args)
    


