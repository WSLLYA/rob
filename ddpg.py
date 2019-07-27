# coding=utf-8

import tensorflow as tf
import numpy as np
from memory import Memory
from priority_memory import PrioritizedMemory
from mpi_running_mean_std import RunningMeanStd        # update the mean and std dynamically.
import tensorflow.contrib as tc
import tf_util as tf_util
from mpi_adam import MpiAdam
from functools import partial


conv2_ = partial(tc.layers.conv2d, kernel_size=3, stride=2, padding='valid', activation_fn=None)
bn = partial(tc.layers.batch_norm, scale=True, updates_collections=None)
def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


# --------------- hyper parameters -------------

LR_A = 0.0001       # learning rate for actor
LR_C = 0.001        # learning rate for critic
GAMMA = 0.99        # reward discount
TAU = 0.005         # soft replacement

# -----------------------------  DDPG ---------------------------------------------

class DDPG(object):
    def __init__(self, memory_capacity, batch_size, prioritiy, alpha=0.2, use_n_step=False, n_step_return=5):
        self.batch_size = batch_size
        self.is_prioritiy = prioritiy
        self.n_step_return = n_step_return
        self.use_n_step = use_n_step
        if prioritiy:
            self.memory = PrioritizedMemory(capacity=memory_capacity, alpha=alpha)
        else:
            self.memory = Memory(limit=memory_capacity, action_shape=(4,),
                             observation_shape=(224, 224, 3),
                             full_state_shape=(24, ))
        self.pointer = 0                        # memory 计数器　
        self.sess = tf.InteractiveSession()     # 创建一个默认会话
        self.lambda_1step = 0.5                 # 1_step_return_loss的权重
        self.lambda_nstep = 0.5                 # n_step_return_loss的权重
        self.beta = 0.6

        # 定义 placeholders
        self.observe_Input = tf.placeholder(tf.float32, [None, 128, 128, 3], name='observe_Input')
        self.observe_Input_ = tf.placeholder(tf.float32, [None, 128, 128, 3], name='observe_Input_')
        self.f_s = tf.placeholder(tf.float32, [None, 24], name='full_state_Input')
        self.f_s_ = tf.placeholder(tf.float32, [None, 24], name='fill_state_Input_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        self.n_step_steps = tf.placeholder(tf.float32, shape=(None, 1), name='n_step_reached')

        with tf.variable_scope('obs_rms'):
            self.obs_rms = RunningMeanStd(shape=(128, 128, 3))
        with tf.variable_scope('state_rms'):
            self.state_rms = RunningMeanStd(shape=(24,))
        with tf.name_scope('obs_preprocess'):
            self.normalized_observe_Input = tf.clip_by_value(
                normalize(self.observe_Input, self.obs_rms), -10., 10.)
            self.normalized_observe_Input_ = tf.clip_by_value(
                normalize(self.observe_Input_, self.obs_rms), -10., 10.)
        with tf.name_scope('state_preprocess'):
            self.normalized_f_s0 = normalize(self.f_s, self.state_rms)
            self.normalized_f_s1 = normalize(self.f_s_, self.state_rms)

        with tf.variable_scope('Actor'):
            self.action = self.build_actor(self.normalized_observe_Input, scope='eval', trainable=True)
            self.action_ = self.build_actor(self.normalized_observe_Input_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            q = self.build_critic(self.normalized_f_s0, self.action, scope='eval', trainable=True)
            q_ = self.build_critic(self.normalized_f_s1, self.action_, scope='target', trainable=False)

        # Collect networks parameters. It would make it more easily to manage them.
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        with tf.variable_scope('Soft_Update'):
            self.soft_replace_a = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.at_params, self.ae_params)]
            self.soft_replace_c = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.ct_params, self.ce_params)]
        with tf.variable_scope('Critic_Lose'):
            self.q_target = self.R + (1. - self.terminals1) * GAMMA * q_

            if self.use_n_step:
                self.n_step_target_q = self.R + (1. - self.terminals1) * tf.pow(GAMMA, self.n_step_steps) * q_
            self.td_error = tf.abs(self.q_target - q)

            if self.use_n_step:
                self.nstep_td_error = tf.abs(self.n_step_target_q - q)

            L2_regular = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.001),
                                                                     weights_list=self.ce_params)
            one_step_losse = tf.reduce_mean(tf.multiply(self.ISWeights, tf.square(self.td_error))) * self.lambda_1step

            if self.use_n_step:
                n_step_td_losses = tf.reduce_mean(tf.multiply(self.ISWeights, self.nstep_td_error)) * self.lambda_nstep
                critic_losses = one_step_losse + n_step_td_losses + L2_regular
            else:
                critic_losses = one_step_losse + L2_regular

        tf.summary.scalar('critic_losses', critic_losses)
        with tf.variable_scope('Actor_lose'):
            a_loss = - tf.reduce_mean(q)

        # Setting optimizer for Actor and Critic
        with tf.variable_scope('Critic_Optimizer'):
            self.critic_grads = tf_util.flatgrad(critic_losses, self.ce_params)
            self.critic_optimizer = MpiAdam(var_list=self.ce_params, beta1=0.9, beta2=0.999, epsilon=1e-08)

        with tf.variable_scope('Actor_Optimizer'):
            self.actor_grads = tf_util.flatgrad(a_loss, self.ae_params)
            self.actor_optimizer = MpiAdam(var_list=self.ae_params, beta1=0.9, beta2=0.999, epsilon=1e-08)

        self.sess.run(tf.global_variables_initializer())
        self.critic_optimizer.sync()

        #  init_target net-work with evaluate net-params
        self.init_a_t = [tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)]
        self.init_c_t = [tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)]
        self.sess.run(self.init_a_t)
        self.sess.run(self.init_c_t)

        # 保存模型
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
        self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.merged_summary = tf.summary.merge_all()

    def choose_action(self, obs):
        obs = obs.astype(dtype=np.float32)
        return self.sess.run(self.action, {self.observe_Input: obs[np.newaxis, :]})[0]

    def Save(self):
        # 只存权重,不存计算图.
        self.saver.save(self.sess, save_path="model/model.ckpt", write_meta_graph=False)

    def load(self):
        self.saver.restore(self.sess, save_path="model/model.ckpt")

    def learn(self):
        if self.is_prioritiy:
            batch, n_step_batch, percentage = self.memory.sample_rollout(
                batch_size=self.batch_size,
                nsteps=self.n_step_return,
                beta=self.beta,
                gamma=GAMMA
            )
        else:
            batch = self.memory.sample(batch_size=self.batch_size)

        one_step_target_q = self.sess.run(
            self.q_target,
            feed_dict={
                self.observe_Input_: batch['obs1'],
                self.R: batch['rewards'],
                self.terminals1: batch['terminals1'],
                self.f_s_: batch['f_s1']
                       })

        if self.is_prioritiy and self.use_n_step:
            n_step_target_q = self.sess.run(
                self.n_step_target_q,
                feed_dict={self.terminals1: n_step_batch["terminals1"],
                           self.n_step_steps: n_step_batch["step_reached"],
                           self.R:  n_step_batch['rewards'],
                           self.observe_Input_: n_step_batch['obs1'],
                           self.f_s_: n_step_batch['f_s1']
                           })

            td_error, critic_grads, s = self.sess.run(
                [self.td_error, self.critic_grads, self.merged_summary],
                feed_dict={
                    self.q_target: one_step_target_q,
                    self.n_step_target_q: n_step_target_q,
                    self.f_s: batch['f_s0'],
                    self.action: batch['actions'],
                    self.ISWeights: batch['weights']
                })
        else:
            td_error, critic_grads, s = self.sess.run(
                [self.td_error, self.critic_grads, self.merged_summary],
                feed_dict={
                    self.q_target: one_step_target_q,
                    self.f_s: batch['f_s0'],
                    self.action: batch['actions'],
                    self.ISWeights: batch['weights']
                })

        self.critic_optimizer.update(critic_grads, stepsize=LR_C)

        actor_grads = self.sess.run(self.actor_grads, {self.observe_Input: batch['obs0'],
                                                       self.f_s: batch['f_s0']})
        self.actor_optimizer.update(actor_grads, stepsize=LR_A)

        if self.is_prioritiy:
            self.memory.update_priorities(batch['idxes'], td_errors=td_error)

        self.sess.run(self.soft_replace_a)
        self.sess.run(self.soft_replace_c)
        self.writer.add_summary(s)

    def store_transition(self,
                         obs0,
                         action,
                         reward,
                         obs1,
                         full_state0,
                         full_state1,
                         terminal1):
        obs0 = obs0.astype(np.float32)
        obs1 = obs1.astype(np.float32)
        full_state0 = full_state0.astype(np.float32)
        full_state1 = full_state1.astype(np.float32)

        self.memory.append( obs0=obs0,
                            f_s0=full_state0,
                            action=action,
                            reward=reward,
                            obs1=obs1,
                            f_s1=full_state1,
                            terminal1=terminal1)

        self.obs_rms.update(np.array([obs0]))
        self.obs_rms.update(np.array([obs1]))

        self.state_rms.update(np.array([full_state0]))
        self.state_rms.update(np.array([full_state1]))

        self.pointer += 1

    def build_actor(self, observe_input, scope, trainable, is_training=True):
        bn_a = partial(bn, is_training=is_training)
        fc_a = partial(tf.layers.dense, activation=None, trainable=trainable)
        conv2_a = partial( conv2_, trainable=trainable)
        relu = partial(tf.nn.relu)
        with tf.variable_scope(scope):
            # conv -> BN -> relu
            net = relu(bn_a(conv2_a( observe_input, 32 )))
            net = relu(bn_a(conv2_a( net, 32 )))
            net = relu(bn_a(conv2_a( net, 64 )))
            net = relu(bn_a(conv2_a( net, 64 )))
            net = relu(bn_a(conv2_a( net, 128 )))
            net = relu(bn_a(conv2_a( net, 128 )))

            net = tf.layers.flatten(net)

            net = relu(bn_a(fc_a( net, 128 )))
            net = relu(bn_a(fc_a( net, 128 )))
            action_output = fc_a( net, 4, activation=tf.nn.tanh,
                                  kernel_initializer=tf.initializers.random_uniform(minval=-0.0003,
                                                                                    maxval=0.0003))
            #输出(1,4)
            action_output = action_output * np.array([0.05, 0.05, 0.05, np.radians(90)])
            # dx a[0] (-0.05,0.05)
            # dy a[1] (-0.05,0.05)
            # dz a[2] (-0.05,0.05)
            # da a[3] (-pi/2,pi/2)

            return action_output

    def build_critic(self, f_s, a, scope, trainable, is_training=True):
        bn_a = partial(bn, is_training=is_training)
        relu = partial(tf.nn.relu)
        fc_c = partial(tf.layers.dense, activation=None, trainable=trainable)
        with tf.variable_scope(scope):

            net = tf.concat([f_s, a], axis=1)
            net = relu(bn_a(fc_c( net, 128 )))
            net = relu(bn_a(fc_c( net, 128 )))

            q = fc_c(net, 1, kernel_initializer=tf.initializers.random_uniform(minval=-0.0003,
                                                                               maxval=0.0003))
            # Q(s,a) 输出一个[None,1]
            return q
