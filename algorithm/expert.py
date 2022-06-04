import sys
sys.path.append("..")
from env.KukaGymEnv import KukaDiverseObjectEnv
from My_toolkit import wreplay
import random
import pdb


def collect(env, epochs=1, steps=1):
    trans = []

    obs = env.reset()
    full_state = env._low_dim_full_state()

    for epoch in range(epochs):
        for step in range(steps):
            pdb.set_trace()
            obs_pos = env.get_ob_pos()
            action = obs_pos[random.randint(0, len(obs_pos)-1)]
            new_obs, reward, done, info = env.step(action)
            new_full_state = env._low_dim_full_state()

            item = [obs, full_state, action, reward, new_obs, new_full_state, done]
            trans.append(item)

            obs = new_obs
            full_state = new_full_state

    wreplay(trans, 'outs/experts.txt')        


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
    pdb.set_trace()
    collect(env)

if __name__ == '__main__':
    main()