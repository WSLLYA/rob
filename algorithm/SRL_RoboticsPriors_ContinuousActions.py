"""
This is the code associated to the paper "Low Dimensional State Representation Learning with Robotics Priors in Continuous Action Spaces"
Botteghi N., et al, published in the International Conference of Intelligent Systems and Robots (IROS), September 2021.

The StateRepresentation class includes the encoder neural network for learning a low-dimensional state representation
from high-dimesional observations (lidar data points + RGB camera images). The encoder is trained with a new set of
robotics priors tailored for continuous state and action spaces.
"""

import tensorflow as tf
# Hide some depreacation warnings and disable eager execution
tf.logging.set_verbosity(tf.logging.ERROR)
import os
import numpy as np
import sonnet as snt
from utils import load_pickle, reshape_observation, save_pickle
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size

        self.obs_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.rews_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.sample_nr_buf = np.zeros(int(size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)

    def store(self, obs, act, rew, done, sample_nr):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.sample_nr_buf[self.ptr] = sample_nr
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)

    # def _get_act_seq

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size - 1, size=batch_size)
        idxs = idxs[self.done_buf[idxs] != 1.]  # remove the last samples of the sequence

        obs_dict = dict(obs=self.obs_buf[idxs],
                        acts=self.acts_buf[idxs],
                        rews=self.rews_buf[idxs - 1],
                        done=self.done_buf[idxs],
                        sample_nr=self.sample_nr_buf[idxs])
        next_obs_dict = dict(obs=self.obs_buf[idxs + 1],
                             acts=self.acts_buf[idxs + 1],
                             rews=self.rews_buf[idxs],
                             done=self.done_buf[idxs + 1],
                             sample_nr=self.sample_nr_buf[idxs + 1])
        return obs_dict, next_obs_dict

    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    rews=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size],
                    sample_nr=self.sample_nr_buf[:self.size])

    def remove_all(self):
        self.__init__(self.obs_dim, self.act_dim, self.size)


class StateRepresentation(object):
    """This class takes care of learning the state representation"""

    def __init__(self, obs_dim, state_dim=128, act_dim=4, batch_size=64, learning_rate=5e-4,
                 alpha=2, beta=10, seed=1, continuelearning=False, usingros=False):

        # folder = "training_results"
        # filename = 'SRLnetwork'
        # if usingros:
        #     import rospkg
        #     self.model_path = os.path.join(rospkg.RosPack().get_path("rosbot_srl"), folder, "srl", filename + ".ckpt")
        # else:
        #     self.model_path = os.path.join("..", folder, "srl", filename + ".ckpt")

        # RGB(128*128*3)
        self.obs_dim = obs_dim   # size [32, 24, 40] with 32x24x3 --> size RGB image and 40 --> size of lidar array
        self.obs_count = obs_dim[0] * obs_dim[1] * 3  #  calculate the flattened dimension of the observation vector
        self.state_dim = state_dim
        self.act_dim = act_dim   # linear and angular velocity of the robot
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha   # priors coefficient (see paper for more details)
        self.beta = beta   # priors coefficient (see paper for more details)
        self.continuelearning = continuelearning

        self.memory = ReplayBuffer(self.obs_count, act_dim=act_dim, size=3e4)

        tf.random.set_random_seed(seed=seed)
        np.random.seed(seed=seed)

        # DEFINE THE FUNCTIONS REQUIRED FOR TRAINING
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define the placeholders
            self.obs_1 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_1')
            self.obs_2 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_2')
            self.obs_3 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_3')
            self.obs_4 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_4')

            self.act_1 = tf.placeholder(tf.float32, shape=[None, 2], name='action_1')
            self.act_2 = tf.placeholder(tf.float32, shape=[None, 2], name='action_2')

            self.is_training = tf.placeholder(tf.bool, shape=[], name="train_cond")

            # Define the neural network architecture and its output
            self.nn = snt.Module(self.SRLencoder, name='SRL_Network')
            self.state_1 = self.nn(self.obs_1, self.is_training)
            self.state_2 = self.nn(self.obs_2, self.is_training)

            self.state_delta = self.state_2 - self.state_1

            self.state_shuff = self.nn(self.obs_3, self.is_training)
            self.state_4 = self.nn(self.obs_4, self.is_training)
            self.state_delt_shuff = self.state_4 - self.state_shuff

            # define losses (i.e. the robotics priors) and optimizer (ADAM)

            self.temp_coh_loss = self.temporal_coherence_prior(self.state_delta, self.act_1,alpha=self.alpha)

            self.caus_loss = self.causality_prior(self.state_1, self.state_shuff, self.act_1,self.act_2, self.beta)

            self.prop_loss = self.proportionality_prior(self.state_delta, self.state_delt_shuff, self.act_1, self.act_2,
                                                        self.beta)

            self.repeat_loss = self.repeatability_prior(self.state_1, self.state_shuff, self.state_delta,
                                                        self.state_delt_shuff, self.act_1, self.act_2, self.beta)

            graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_regularization_loss = tf.reduce_sum(graph_regularizers)

            self.losses = [0, 1*self.temp_coh_loss, 2*self.caus_loss, 1*self.prop_loss, 1*self.repeat_loss,
                           total_regularization_loss]
            self.loss = tf.reduce_sum(self.losses)
            self.losses[0] = tf.reduce_sum(self.losses[:])

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss)

            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

        # Initialize the session
        self.sess = tf.Session(graph=self.graph)
        # Restore session if desired
        if continuelearning:
            print('Loading SR model from memory')
            self.load_model()
        else:
            print("Starting SR model from scratch!")
            self.sess.run(self.init)

    def SRLencoder(self, observations, is_training, l2_reg=0.001, batch_norm=False):

        """
        The camera image and the lidar data points are separated from the flattened observation vector via snt.SliceByDim()
        in camera_inputs and laser_inputs. camera_inputs are reshaped into [32, 24, 3] and fed to conv2d layers,
        while laser_inputs are reshaped into [40, 1] and fed to a conv1d layer. Noise is added to the output during training
        for preventing equal state predictions (for more details we reference to "Learning State Representations with Robotics Priors"
        Jonschkowski R. et al).

         image               laser
           |                   |
         conv2d              conv1d
           |                   |
         conv2d                |
           |                   |
        flatten             flatten
           |                   |
         dense               dense
           |                   |
           --------merge--------
                     |
                   dense
                     |
                   dense
                     |
                   dense
                     |
                   state


        :param observations:
        :param is_training:
        :param l2_reg:
        :param batch_norm:
        :return state:
        """

        regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=l2_reg)}
        initializers = {"w": tf.keras.initializers.he_normal()}

        # Camera branch
        camera_inputs = snt.SliceByDim(dims=[1], begin=[0], size=[self.obs_dim[0] * self.obs_dim[1] * 3])(observations)
        camera_inputs = tf.reshape(camera_inputs, [-1, self.obs_dim[0], self.obs_dim[1], 3])

        camera_conv1 = tf.layers.conv2d(camera_inputs, filters=32, kernel_size=3, strides=1, padding='valid',
                                        activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                        kernel_initializer=initializers["w"])

        if batch_norm:
            camera_conv1 = tf.layers.batch_normalization(camera_conv1, training=is_training)

        camera_conv2 = tf.layers.conv2d(camera_conv1, filters=64, kernel_size=3, strides=1, padding='valid',
                                        activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                        kernel_initializer=initializers["w"])
        if batch_norm:
            camera_conv2 = tf.layers.batch_normalization(camera_conv2, training=is_training)

        camera_flatten = snt.BatchFlatten()(camera_conv2)

        camera_dense1 = tf.layers.dense(camera_flatten, 64, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                        kernel_initializer=initializers["w"])

        # Laser branch
        laser_inputs = snt.SliceByDim(dims=[1], begin=[self.obs_dim[0] * self.obs_dim[1] * 3], size=[self.obs_dim[2]])(
            observations)
        laser_inputs = tf.reshape(laser_inputs, [-1, self.obs_dim[2], 1])

        laser_conv1 = tf.layers.conv1d(laser_inputs, filters=32, kernel_size=3, strides=1, padding='valid',
                                       activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])

        if batch_norm:
            laser_conv1 = tf.layers.batch_normalization(laser_conv1, training=is_training)


        laser_flatten = snt.BatchFlatten()(laser_conv1)

        laser_dense1 = tf.layers.dense(laser_flatten, 64, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])

        # Merged layers
        merge_input = tf.concat(values=[camera_dense1, laser_dense1], axis=1)
        merge_dense1 = tf.layers.dense(merge_input, 64, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])
        merge_dense2 = tf.layers.dense(merge_dense1, 32, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])

        state = tf.layers.dense(merge_dense2, self.state_dim, activation=None,
                                      kernel_regularizer=regularizers["w"], kernel_initializer=tf.initializers.zeros)

        merge_noise = lambda x: tf.cond(self.is_training, lambda: x + tf.random_normal(shape=tf.shape(x), stddev=1e-6),
                                        lambda: x)
        return merge_noise(state)

    def remember(self, observation, action, reward, done, sample_number):
        self.memory.store(observation, action, reward, done, sample_number)

    def learn(self):
        with tqdm(total=self.memory.size) as pbar:
            t_loss = 0
            losses = np.zeros(6)
            nr_batches = self.memory.size // self.batch_size
            for _ in range(nr_batches):
                b1 = self.memory.sample_batch(self.batch_size)
                b2 = self.memory.sample_batch(self.batch_size)

                l = min(len(b1[0].get('obs')), len(b2[0].get('obs')))

                feed_dict = {self.obs_1: b1[0].get('obs')[:l], self.obs_2: b1[1].get('obs')[:l],
                             self.obs_3: b2[0].get('obs')[:l], self.obs_4: b2[1].get('obs')[:l],
                             self.act_1: b1[0].get('acts')[:l],
                             self.act_2: b2[0].get('acts')[:l],
                             self.is_training: True}

                _, loss, l = self.sess.run([self.train_op, self.loss, self.losses], feed_dict=feed_dict)
                t_loss += loss
                losses += l
                pbar.update(self.batch_size)

        return t_loss / nr_batches, losses / nr_batches

    def predict(self, observation):
        feed_dict = {self.obs_1: np.reshape(observation, (1, -1)),
                     self.is_training: False}
        state = self.sess.run(self.state_1, feed_dict=feed_dict)
        return state

    def predict_all(self, observations, batch_size=512):
        observations = observations.reshape(len(observations), -1)
        states = np.ndarray([len(observations), self.state_dim])
        num_batches = int(np.trunc((observations.shape[0] / batch_size)))

        for i in range(num_batches):
            states[i * batch_size: (i + 1) * (batch_size)] = self.sess.run(self.state_1, feed_dict={
                self.obs_1: observations[i * batch_size: (i + 1) * (batch_size)], self.is_training: False})
        states[num_batches * batch_size:] = self.sess.run(self.state_1, feed_dict={
            self.obs_1: observations[num_batches * batch_size:], self.is_training: False})
        return states

    def get_memory_states(self):
        states = []
        mem = self.memory.get_all_samples().get('obs')

        states = self.predict_all(mem)
        return states, self.memory.get_all_samples()

    def save_model(self):
        print("SRL Network storing model..........")
        return self.saver.save(self.sess, self.model_path)

    def load_model(self):
        print("SRL Network restoring data...........")
        self.saver.restore(self.sess, self.model_path)

    # Each prior is defined in a function

    def temporal_coherence_prior(self, s_d, a1, alpha=2):
        """
        E[||s_t+1 - s_t||_2 * e^(-alpha * ||a||_2)]

        :param s_d: s_t+1 - s_t
        :param a1:  action connecting s_t and s_t+1
        :param alpha:
        :return: expectation of the temporal coherence loss
        """

        return tf.reduce_mean(tf.math.exp(-alpha * tf.norm(a1, ord=2, axis=1)) * tf.norm(s_d, ord=2, axis=1) ** 2)


    def causality_prior(self, s1, s2, a1, a2, beta):
        """
        E[e^(-||s1-s2||^2) * e^(-beta*||a1-a2||^2)]

        :param s1: state s_t1
        :param s2: state s_t2
        :param a1: action connecting s_t1 and s_t1+1
        :param a2:  action connecting s_t2 and s_t2+1
        :param beta: weighting factor
        :return: expectation of the causality loss
        """
        closs1 = tf.math.exp(-tf.norm(s1 - s2, ord=2, axis=1) ** 2)
        closs2 = tf.math.exp(-beta*tf.norm(a2 - a1, ord=2, axis=1) ** 2)

        return tf.reduce_mean(closs1 * closs2)

    def proportionality_prior(self, sd1, sd2, a1, a2, beta):
        """
        E[(||s_t2+1 - s_t2||_2 - ||s_t1+1 - s_t1||_2)^2 * e^(-beta * ||a1 - a2||_2)^2]

        :param sd1: s_t1+1 - s_t1
        :param sd2: s_t2+1 - s_t2
        :param a1: action connecting s_t1 and s_t1+1
        :param a2: action connecting s_t2 and s_t2+1
        :param beta: weighting factor
        :return: expectation of the proportinality loss
        """
        ploss1 = (tf.norm(sd2, ord=2, axis=1) - tf.norm(sd1, ord=2, axis=1)) ** 2
        ploss2 = tf.math.exp(-beta * tf.norm(a1 - a2, ord=2, axis=1) ** 2)

        return tf.reduce_mean(ploss1 * ploss2)


    def repeatability_prior(self, s1, s2, s_d1, s_d2, a1, a2, beta):
        """
        E[||(s_t2+1 - s_t2) - (s_t1+1 - s_t1)||_2)^2 * e^(-||s1-s2||^2) * e^(-beta * ||a1 - a2||_2)^2]


        :param s1: state s_t1
        :param s2: state s_t2
        :param s_d1: s_t1+1 - s_t1
        :param s_d2: s_t2+1 - s_t2
        :param a1: action connecting s_t1 and s_t1+1
        :param a2: action connecting s_t2 and s_t2+1
        :param beta: weighting factor
        :return: expectation of the repeatability loss
        """
        rloss1 = tf.math.exp(-tf.norm(s1 - s2, ord=2, axis=1) ** 2)
        rloss2 = tf.norm(s_d2 - s_d1, ord=2, axis=1) ** 2
        rloss3 = tf.math.exp( -beta*tf.norm(a1 - a2, ord=2, axis=1) ** 2)

        return tf.reduce_mean(rloss1 * rloss2 * rloss3)


if __name__ == '__main__':
    from Logger import Logger
    import Plotter as plotter

    # Load saved observations from memory (samples collected by random exploring the large 4 walls environment --> see our paper)
    folder = 'training_data/observations_4walls_large.pkl'

    data = load_pickle(folder)
    print('Loaded {} data points'.format(len(data)))


    srl = StateRepresentation(obs_dim=[32, 24, 40], state_dim=5, act_dim=2, batch_size=256, learning_rate=5e-4, alpha=2,
                              beta=10, seed=3, continuelearning=False, usingros=False)


    # add all data to srl memory
    for d in data:
        srl.remember(reshape_observation(d[0]), d[1], d[2], d[3], d[4])

    epochs = 20
    loss_history = []

    # training the encoder for 20 epochs
    for epoch in range(epochs):
        loss, losses = srl.learn()
        print('Finished epoch {}/{}. The loss this epoch was: {}'.format(epoch + 1, epochs, loss))
        print('temp_coh_loss: {} caus_loss: {} prop_loss: {} repeat_loss: {}'.format(losses[1], losses[2], losses[3],
                                                                                     losses[4]))

        loss_history.append(losses)

    # save the trained model
    srl.save_model()

    # compute all the states from the observations
    print('Finished training, now predicting all states')
    lg = Logger('./')
    gt = []
    for d in data:
        gt.append(d[0][40:45])
        lg.log("position_obs", d[0][40:45])
        lg.log("rewards", d[2])

    states, _ = srl.get_memory_states()

    # visualise the results
    print('Plotting the results')

    plotter.plot_all_srl(states=states, rews=srl.memory.get_all_samples().get('rews'), data_dict=lg.logDict,
                         loss_history=loss_history, trainingcycle=2, save=True, size=15000, name="continuous")


