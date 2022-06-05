from email import policy
import sys, os
sys.path.append("..")
from env.KukaGymEnv import KukaDiverseObjectEnv
from gym import spaces
from My_toolkit import wreplay
import random

import numpy as np
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


def collect(env, policy, epochs=1, steps=10):
    trans = []

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

            item = [np.array(obs), np.array(full_state),
                    np.array(action), np.array(reward),
                    np.array(new_obs), np.array(new_full_state), np.array(done)]
            trans.append(item)

            obs = new_obs
            full_state = new_full_state

    path = os.path.dirname(os.getcwd())
    path = os.path.join(path, 'outs/experts.txt')
    wreplay(trans[0], path)   


def main():
    env = KukaDiverseObjectEnv(renders=True,
                               isDiscrete=False,
                               maxSteps=20,
                               blockRandom=0.4,
                               removeHeightHack=True,
                               use_low_dim_obs=False,
                               use_segmentation_Mask=False,
                               numObjects=4,
                               dv=1.0)
    policy = ContinuousDownwardBiasPolicy()
    pdb.set_trace()
    collect(env, policy)

if __name__ == '__main__':
    main()