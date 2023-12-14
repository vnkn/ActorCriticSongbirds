import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


BATCH_SIZE = 4
N_TONES = 5
LEARNING_RATE = 0.5
ENTROPY_COEF = 0.0


def sigmoid(x):
  return 1/(1+np.exp(-x))


def make_random_var():
    return tf.Variable(np.random.normal())


def get_rewards(sample, target):
    # return 1 - (sample - target) ** 2.
    return 2. * (sample == target) - 1.


class Critic:
    """The critic model"""
    def __init__(self, song_length, lesioned=False):
        self.song_length = song_length
        self.lesioned = lesioned
        self.raw_values = tf.Variable(np.zeros(shape=(self.song_length, 1)),
            dtype=tf.float32)
        self.values = self.raw_values

    def get_loss_and_train_op(self, true_values):
        timestep_losses = (self.values - true_values) ** 2.
        loss = tf.reduce_mean(timestep_losses)
        if self.lesioned:
            train_op = tf.identity(-1)
        else:
            train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        return timestep_losses, train_op

    def get_values(self):
        return self.values


class Actor:
    """The actor model"""
    def __init__(self, song_length):
        self.song_length = song_length

        self.logits = tf.Variable(np.ones(shape=(song_length, N_TONES)))
        self.normalization = tf.log(tf.reduce_sum(tf.exp(self.logits)))
        self.log_prob = self.logits - self.normalization

        self.dist = tf.nn.softmax(tf.nn.relu(self.logits), axis=1)
        self.actions = tf.random.categorical(self.dist, num_samples=1)

    def get_actions(self, sess):
        dist = sess.run(self.dist)
        return np.array([
                np.random.choice(np.arange(N_TONES), p=dist[i,:].ravel())
                for i in range(dist.shape[0])
            ]).reshape((self.song_length, 1))
        # return self.actions

    def get_loss_and_train_op(self, advantages, actions):
        # entropy_terms = ENTROPY_COEF * self.distributions.entropy() * -1.
        indices = tf.transpose(tf.stack((
            np.arange(self.song_length).reshape((self.song_length,1)),
            actions)))
        losses = advantages * tf.gather_nd(self.log_prob, indices) * -1.
        loss = tf.reduce_mean(losses)
        return loss, tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


class Model:
    """The songbird actor-critic model"""
    def __init__(self, song_length, lesion_critic=False, target_diversity=0.):
        self.song_length = song_length
        self.target_diversity = target_diversity
        self.critic = Critic(song_length, lesioned=lesion_critic)
        self.actor = Actor(song_length)

        self.true_values_ph = tf.placeholder(tf.float32, shape=(song_length,1))
        self.actions_ph = tf.placeholder(tf.int32, shape=(song_length,1))
        self.advantages_ph = tf.placeholder(tf.float64, shape=(song_length,1))

        self.critic_loss, self.critic_train_op = self.critic.get_loss_and_train_op(self.true_values_ph)
        self.actor_loss, self.actor_train_op = self.actor.get_loss_and_train_op(self.advantages_ph, self.actions_ph)

    def train(self, target, sess, off_policy=False):
        if not off_policy:
            actions = self.actor.get_actions(sess)
        else:
            actions = np.random.randint(0, N_TONES, size=(self.song_length, 1))
        rewards = get_rewards(actions, target)
        mean_reward = np.mean(rewards)
        if mean_reward >= (1. - self.target_diversity):
            return 0., 0., mean_reward

        c_loss, _ = sess.run([self.critic_loss, self.critic_train_op], {self.true_values_ph: rewards})
        predicted_values = sess.run(self.critic.get_values())

        advantages = rewards - predicted_values

        # print("Target = ", target[0])
        # print("Action = ", actions[0])
        # print("Dist = ", sess.run(self.actor.dist))
        # print("Reward = ", rewards[0])
        # print("Pred Value = ", predicted_values[0])
        # print("Adv = ", advantages[0])

        a_loss, _ = sess.run(
            [self.actor_loss, self.actor_train_op],
            {
                self.actions_ph: actions,
                self.advantages_ph: advantages,
            })
        return c_loss, a_loss, mean_reward


# def test():
#     song_length = 10
#     m = Model(song_length, lesion_critic=True)
#     target = np.random.randint(0, N_TONES, size=(song_length,1))
#     rews = []
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())

#         for i in range(500):
#             print("\nITERATION {}\n".format(i))
#             c_loss, a_loss, total_reward = m.train(target, sess)
#             rews.append(total_reward)
#             print(total_reward)
        
#     plt.plot(rews)
#     plt.show()

# test()
