import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

N_TONES = 6
LEARNING_RATE = 0.2
EPSILON = 0.5

def sigmoid(x):
  return 1/(1+np.exp(-x))


def make_random_var():
    return tf.Variable(np.random.normal())


def get_rewards(sample, target):
    # return 1 - (sample - target) ** 2.
    return 2. * (sample == target) - 1.


class Critic:
    """The critic model"""
    def __init__(self, song_length):
        self.song_length = song_length
        self.raw_values = tf.Variable(np.zeros(shape=(self.song_length, 1)),
            dtype=tf.float32)
        self.values = self.raw_values

    def get_loss_and_train_op(self, true_values,learning_rate):

        loss = tf.reduce_mean((self.values - true_values) ** 2.)
        return loss, tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def get_values(self):
        return self.values


class Actor:
    """The actor model"""
    def __init__(self, song_length):
        self.song_length = song_length


        arr = np.ones(shape=(song_length, N_TONES))
        self.logits = tf.Variable(np.ones(shape=(song_length, N_TONES)))
        # Change the Logits to be randomized, and to use an epsilon random policy
        for k in range(len(arr)):
            for l in range(len(arr[k])):
                arr[k][l] *= np.random.random_sample()
                #print(arr[k][l])
            #a = arr * np.random.random_sample()
        #self.logits = tf.Variable(np.random.rand(shape=(song_length, N_TONES)))

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

    def get_loss_and_train_op(self, advantages, actions,learning_rate):
        # entropy_terms = ENTROPY_COEF * self.distributions.entropy() * -1.
        indices = tf.transpose(tf.stack((
            np.arange(self.song_length).reshape((self.song_length,1)),
            actions)))
        losses = advantages * tf.gather_nd(self.log_prob, indices) * -1.
        loss = tf.reduce_mean(losses)
        return loss, tf.train.AdamOptimizer(learning_rate).minimize(loss)


class Model:
    """The songbird actor-critic model"""
    def __init__(self, song_length, learning_rate):
        self.song_length = song_length
        self.critic = Critic(song_length)
        self.actor = Actor(song_length)
        self.epsilon = EPSILON

        self.true_values_ph = tf.placeholder(tf.float32, shape=(song_length,1))
        self.actions_ph = tf.placeholder(tf.int32, shape=(song_length,1))
        self.advantages_ph = tf.placeholder(tf.float64, shape=(song_length,1))

        self.critic_loss, self.critic_train_op = self.critic.get_loss_and_train_op(self.true_values_ph, learning_rate)
        self.actor_loss, self.actor_train_op = self.actor.get_loss_and_train_op(self.advantages_ph, self.actions_ph,learning_rate)

    def train(self, target, sess, epsilon = EPSILON):
        # Epsilon Greedy Policy for Taking Actions
        # Generate a Random Number, if your number is  < Epsilon then take a random action.
        number = np.random.rand()
        if(number < epsilon):
            actions =  np.random.random()
        #
        actions = self.actor.get_actions(sess)
        rewards = get_rewards(actions, target)

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
        return c_loss, a_loss, np.mean(rewards)


def test(learning_rate = 0.5):
    song_length = 10
    m = Model(song_length,learning_rate)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    print( "Target is " )
    print(target)
    rews = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(5000):
            #print("\nITERATION {}\n".format(i))
            c_loss, a_loss, total_reward = m.train(target, sess)
            rews.append(total_reward)
            #print(total_reward)
    plt.xlabel('iteration')
    plt.ylabel('Reward')
    plt.plot(rews)
    plt.show()
def test_epsilons(learning_rate = 0.5):
    song_length = 10
    m = Model(song_length,learning_rate)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    print( "Target is " )
    print(target)
    rews = []
    iters = []
    epsilons = np.linspace(0,.8,9)
    for epsilon in epsilons:
        sum = 0
        for trial in range(5):
            print(epsilon)
            done = False
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                for i in range(5000):
                    #print("\nITERATION {}\n".format(i))
                    c_loss, a_loss, total_reward = m.train(target, sess, epsilon)
                    if(total_reward > 0.9):
                        #iters.append(i)
                        sum = sum + i
                        done = True
                        break 
                if(done == False):
                    sum = sum + 1000
        sum = sum/5
        iters.append(sum)
                    #print(total_reward)
    plt.xlabel('Epsilon Value')
    plt.ylabel('Number of Iterations')
    plt.plot(epsilons, iters)
    plt.show()
    print("hi")
    plt.savefig('image.png')
def test_learning_rates(learning_rate):
    song_length = 10
    m = Model(song_length, learning_rate)
    target = np.random.randint(0, N_TONES, size=(song_length,1))
    learning_rates = {}
    rews = []
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(50):
            #print("\nITERATION {}\n".format(i))
            c_loss, a_loss, total_reward = m.train(target, sess)
            rews.append(total_reward)
    return rews[49]
def sim_learning_rates():
    learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    avgs = []
    for learning_rate in learning_rates:
        sum = 0
        for i in range(10):
            print(i)
            sum += test_learning_rates(learning_rate)
        avgs.append(sum)  
    print(avgs)
    plt.plot(avgs)
    plt.show()
    plt.xlabel('Learning_Rate')
    plt.ylabel('Final Reward')
def plot():
    learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    for learning_rate in learning_rates:
        song_length = 10
        m = Model(song_length,learning_rate)
        target = np.random.randint(0, N_TONES, size=(song_length,1))
        print( "Target is " )
        print(target)
        rews = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(500):
                #print("\nITERATION {}\n".format(i))
                c_loss, a_loss, total_reward = m.train(target, sess)
                rews.append(total_reward)
                #print(total_reward)
        plt.plot(rews, label = learning_rate)
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.show()
    plt.savefig('Iteration_Reward.png')

def run():
    #test()
    #sim_learning_rates()
    #plot()
    test_epsilons()
run()
# Right now it's not an epsilon random policy (Works very well because it's uniform)
# instead, if we randomize this, we could make it random
# 
