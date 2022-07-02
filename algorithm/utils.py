import numpy as np
try:
    import cPickle as pickle
except:
    import pickle


def flatten_list(l):
	return [item for sublist in l for item in sublist]


def reshape_observation(observation):
    """
    reshape observations coming from the ROS/Gazebo simulations. The observation is composed of 40 lidar data points,
    goal position (xg, yg), robot pose (x, y theta) and 1 RGB image

    :param observation:
    :return: reshaped observation
    """
    observation = np.array(observation)
    obs_laser = np.array(observation[:40] * 100).astype(np.uint8)
    obs_laser = obs_laser.reshape(1,len(obs_laser))
    obs_camera = np.array(observation).astype(np.uint8)[45:]
    obs_camera = obs_camera.reshape(1,len(obs_camera))
    observation = np.concatenate((obs_camera, obs_laser), axis=1).reshape(1, len(observation) - 5).astype(np.float32) # the goal position (xg, yg) and the robot pose (x, y theta) are discarted as assumed not available
    return observation

def reshape_action(action):
    action = np.array(action)
    action = action.reshape((1,))
    action = np.column_stack((action, action.shape))

    return action

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    if not file[-3:] == 'pkl' and not file[-3:] == 'kle':
        file = file+'pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data

