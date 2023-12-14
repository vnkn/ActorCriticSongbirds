import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


BATCH_SIZE = 4
LEARNING_RATE = 0.1
ENTROPY_COEF = 0.0


def sigmoid(x):
  return 1/(1+np.exp(-x))


def make_random_var():
    return tf.Variable(np.random.normal())


def get_rewards(sample, target):
    n = sample.shape[1]
    return np.ones(n) - (sample - target) ** 2.


class Critic:
    """The critic model"""
    def __init__(self, song_length, lesioned=False):
        self.song_length = song_length
        self.lesioned = lesioned
        self.values = tf.sigmoid([make_random_var() for _ in range(self.song_length)])

    def get_loss_and_train_op(self, true_values):
        loss = tf.reduce_mean((self.values - true_values) ** 2.)
        if self.lesioned:
            train_op = tf.identity(3)
        else:
            train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        return loss, train_op

    def get_values(self):
        return self.values


class Actor:
    """The actor model"""
    def __init__(self, song_length):
        self.song_length = song_length

        self.means = [make_random_var() for _ in range(self.song_length)]
        self.stds = [tf.maximum(tf.Variable(np.random.uniform() + 0.6), 0.1) for _ in range(self.song_length)]
        # self.distributions = [tfd.Normal(mu, sigma) for (mu, sigma) in zip(self.means, self.stds)]
        # self.actions = [dist.sample() for dist in self.distributions]
        self.distributions = tfd.Normal(tf.stack([self.means] * BATCH_SIZE), tf.stack([self.stds] * BATCH_SIZE))
        self.actions = tf.sigmoid(self.distributions.sample())
        # self.actions_batch = tf.stack([self.actions] * BATCH_SIZE)

    def get_actions(self):
        return self.actions

    def get_actions_batch(self):
        return self.actions_batch

    def get_loss_and_train_op(self, advantages, actions):
        entropy_terms = ENTROPY_COEF * self.distributions.entropy() * -1.
        losses = advantages * self.distributions.log_prob(actions + 1e-8) * -1. + entropy_terms
        # for (a, adv, d) in zip(actions, advantages, self.distributions):
        #     losses.append(adv * d.log_prob(a) * -1.)
        # losses = tf.map_fn(lambda e: e[1] * e[2].log_prob(e[0]) * -1., list(zip(actions, advantages, self.distributions)))
        loss = tf.reduce_mean(losses)
        # loss = tf.Variable(0.)
        return loss, tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


class Model:
    """The songbird actor-critic model"""
    def __init__(self, song_length, lesion_critic=False):
        self.song_length = song_length
        self.critic = Critic(song_length, lesioned=lesion_critic)
        self.actor = Actor(song_length)

        self.true_values_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, song_length))
        self.actions_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, song_length))
        self.advantages_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, song_length))

        self.critic_loss, self.critic_train_op = self.critic.get_loss_and_train_op(self.true_values_ph)
        self.actor_loss, self.actor_train_op = self.actor.get_loss_and_train_op(self.advantages_ph, self.actions_ph)

    def train(self, target, sess):
        actions = sess.run(self.actor.get_actions())
        rewards = get_rewards(actions, target)

        c_loss, _ = sess.run([self.critic_loss, self.critic_train_op], {self.true_values_ph: rewards})
        predicted_values = sess.run(self.critic.get_values())

        advantages = rewards - predicted_values

        mean, std = sess.run([self.actor.means[0], self.actor.stds[0]])
        mean = sigmoid(mean)
        # print((mean, std))
        # print("Target = ", target[0])
        # print("Action = ", actions[:,0])
        # print("Reward = ", rewards[:,0])
        # print("Pred Value = ", predicted_values[0])
        # print("Adv = ", advantages[:,0])

        a_loss, _ = sess.run(
            [self.actor_loss, self.actor_train_op],
            {
                self.actions_ph: actions,
                self.advantages_ph: advantages,
            })
        return c_loss, a_loss, np.mean(rewards)


def test():
    m = Model(10)
    target = np.random.uniform(size=(10,))
    rews = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(5000):
            # print("\nITERATION {}\n".format(i))
            # print(sess.run(m.actor.actions))
            c_loss, a_loss, total_reward = m.train(target, sess)
            rews.append(total_reward)
            print(total_reward)
        
    plt.plot(rews)
    plt.show()
test()
