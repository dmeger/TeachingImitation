import numpy as np
import random
#from open ai baselines
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, obs_next, action, reward, done):
        data = (obs, obs_next, action, reward, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs_list, obs_list_next, actions_list, rewards_list, dones_list = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, obs_next, action, reward, done = data
            actions_list.append(np.array(action, copy=False))
            rewards_list.append(reward)
            obs_list.append(np.array(obs, copy=False))
            obs_list_next.append(np.array(obs_next, copy=False))
            dones_list.append(done)
        return np.array(obs_list), np.array(obs_list_next), np.array(actions_list), np.array(rewards_list), np.array(dones_list)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        obs_batch_next: np.array
            batch of observations, t+1
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
