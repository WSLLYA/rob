#coding=utf-8
import numpy as np

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape, full_state_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.full_state0 = RingBuffer(limit, shape=full_state_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)
        self.full_state1 = RingBuffer(limit, shape=full_state_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        full_state0_batch = self.full_state0.get_batch(batch_idxs)
        full_state1_batch = self.full_state1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'f_s0': array_min2d(full_state0_batch),
            'f_s1': array_min2d(full_state1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'weights': np.ones(array_min2d(reward_batch).shape), # all 1.
            'demos': np.zeros_like(array_min2d(reward_batch))  # all 0.
        }
        return result

    def append(self, obs0, f_s0, action, reward, obs1, f_s1, terminal1, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.full_state0.append(f_s0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.full_state1.append(f_s1)
        self.terminals1.append(terminal1)

    # testing
    def get_replay(self):
        trans = []
        item = []
        for num in range(self.observations0.__len__()):
            item.append(self.observations0.__getitem__(num))
            item.append(self.full_state0.__getitem__(num))
            item.append(self.actions.__getitem__(num))
            item.append(self.rewards.__getitem__(num))
            item.append(self.observations1.__getitem__(num))
            item.append(self.full_state1.__getitem__(num))
            item.append(self.terminals1.__getitem__(num))
            trans.append(item)
        return trans

    @property
    def nb_entries(self):
        return len(self.observations0)