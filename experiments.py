import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
sns.set(style='darkgrid')

from model_discrete import *


def train_normal(song_length=10):
    m = Model(song_length)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    rews = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            print("\nITERATION {}\n".format(i))
            c_loss, a_loss, total_reward = m.train(target, sess)
            rews.append(total_reward)
            print(total_reward)
        
    plt.plot(rews)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.show()


def train_lesion_critic(song_length=10):
    m = Model(song_length, lesion_critic=True)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    rews = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            print("\nITERATION {}\n".format(i))
            c_loss, a_loss, total_reward = m.train(target, sess)
            rews.append(total_reward)
            print(total_reward)
        
    plt.plot(rews)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.show()


def distortion_error(song_length=10):
    m = Model(song_length)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    rews = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            c_loss, a_loss, total_reward = m.train(target, sess)
            rews.append(total_reward)

        distorted_song = target + np.random.randint(0, 2, size=(song_length,1)) - 1
        regular_loss = get_rewards(distorted_song, target) - sess.run(m.critic.get_values())
        distorted_loss = get_rewards(target, target) - sess.run(m.critic.get_values())
        # distorted_loss = sess.run(m.critic_loss, {m.true_values_ph: get_rewards(distorted_song, target)})

        ax1 = plt.subplot(121)
        plt.bar(np.arange(song_length), distorted_loss.flatten())
        plt.xlabel("Timestep")
        plt.ylabel("Firing Rage")
        plt.title("Distorted Song")

        plt.subplot(122, sharey=ax1)
        plt.bar(np.arange(song_length), regular_loss.flatten())
        plt.xlabel("Timestep")
        plt.ylabel("Firing Rate")
        plt.title("Undistorted Song")

        plt.show()


def train_diversity(song_length=10):
    target_diversity = 0.1
    m = Model(song_length, target_diversity=target_diversity)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    rews = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            print("\nITERATION {}".format(i))
            c_loss, a_loss, total_reward = m.train(target, sess)
            rews.append(total_reward)
            print(total_reward)

    plt.plot(rews)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")


def train_off_policy(song_length=10):
    m = Model(song_length)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    rews = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            print("\nITERATION {}".format(i))
            c_loss, a_loss, total_reward = m.train(target, sess, off_policy=True)
            rews.append(total_reward)
            print(total_reward)
        for i in range(200):
            print("\nITERATION {}".format(1000 + i))
            c_loss, a_loss, total_reward = m.train(target, sess, off_policy=False)
            rews.append(total_reward)
            print(total_reward)

    plt.plot(rews)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.show()
    plt.show()


# train_normal()
# train_lesion_critic()
# distortion_error()
# train_diversity()
train_off_policy()
