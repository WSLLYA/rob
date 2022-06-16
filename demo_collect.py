import random
from gym import spaces
import numpy as np
import argparse
import sys, os
from multiprocessing import Pool

from env.KukaGymEnv import KukaDiverseObjectEnv

import pdb


class ContinuousDownwardBiasPolicy(object):
  """Policy which takes continuous actions, and is biased to move down.
  """

  def __init__(self, height_hack_prob=0.9):
    """Initializes the DownwardBiasPolicy.

    Args:
        height_hack_prob: The probability of moving down at every move.
    """
    self._height_hack_prob = height_hack_prob
    self._action_space = spaces.Box(low=-1, high=1, shape=(4,))

  def sample_action(self, obs, explore_prob):
    """Implements height hack and grasping threshold hack.
    """
    dx, dy, dz, da = self._action_space.sample()
    if np.random.random() < self._height_hack_prob:
      dz = -1
    return [dx, dy, dz, da]

def common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--priority", action="store_true", help="priority memory replay")
    parser.add_argument('--max_ep_steps',    type=int, default=100, help="一个episode最大长度. default = 100")
    return  parser

parser = common_arg_parser()
args = parser.parse_args()

def collect(agent, env, policy, epochs=10, steps=10):

    for epoch in range(epochs):
        obs = env.reset()
        done = False
        full_state = env._low_dim_full_state()

        while not done:
            # obs_pos = env.get_ob_pos()
            # action = np.array(obs_pos[random.randint(0, len(obs_pos)-1)] + (0.3,))
            action = policy.sample_action(obs, 1)
            new_obs, reward, done, info = env.step(action)
            new_full_state = env._low_dim_full_state()

            # item = [np.array(obs), np.array(full_state),
            #         np.array(action), np.array(reward),
            #         np.array(new_obs), np.array(new_full_state), np.array(done)]

            agent.store_transition(obs, action, reward, new_obs, full_state, new_full_state,
                                   done, demo=True)

            obs = new_obs
            full_state = new_full_state

    # path = os.path.dirname(os.getcwd())
    # path = os.path.join(path, 'outs/experts.txt')
    # wreplay(item, path)   

def collect_worker(worker_index):
    print (str(worker_index)+" start!")
    i = worker_index * 300
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=args.max_ep_steps,
                               blockRandom=0.4,
                               removeHeightHack=True,
                               use_low_dim_obs=False,
                               use_segmentation_Mask=False,
                               numObjects=4,
                               dv=1.0)
    while 1:
        obs0, done = env.reset(), False
        f_s0 = env._low_dim_full_state()
        for j in range(args.max_ep_steps):
            action = env.demo_policy()

            obs1, reward, done, info = env.step(action)
            if info['is_success']:
                print('success in %d transition'%(i))
            f_s1 = env._low_dim_full_state()
            demo_transitions = {'obs0': obs0,
                                'f_s0': f_s0,
                                'action': action,
                                'obs1': obs1,
                                'reward': reward,
                                'f_s1': f_s1,
                                'terminal1': done}
            obs0 = obs1
            f_s0 = f_s1

            demo_tran_file_path = 'demos/demo%d.npy'%(i)
            np.save(demo_tran_file_path, demo_transitions, allow_pickle=True)

            i = i + 1
            if done:
                break

            if i >= (worker_index+1) * 300  :
                return

def main():
    # 开12个进程 一起收集demo
    print('Parent process %s.' % os.getpid())
    p = Pool(12)

    for k in range(12):
        p.apply_async(collect_worker, args=(k,))
    p.close()
    p.join()
    print('All subprocesses done.')
    # collect_worker(1)

if __name__ == '__main__':
    main()

    # env = KukaDiverseObjectEnv(renders=True,
    #                            isDiscrete=False,
    #                            maxSteps=1000,
    #                            blockRandom=0.4,
    #                            removeHeightHack=True,
    #                            use_low_dim_obs=False,
    #                            use_segmentation_Mask=False,
    #                            numObjects=1,
    #                            dv=1.0)

    # for i in range (10000):
    #     obs0, done = env.reset(), False
    #     f_s0 = env._low_dim_full_state()
    #     for j in range(100):
    #         action = env.demo_policy()

    #         obs1, reward, done, info = env.step(action)
    #         if info['is_success']:
    #             print('success in %d transition'%(i))
    #         if done:
    #             break
    #         f_s1 = env._low_dim_full_state()
    #         obs0 = obs1
    #         f_s0 = f_s1
    #     if not done:
    #         print("grasp fail!!")