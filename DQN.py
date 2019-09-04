# DQN.py

# set lr 0.01 (from deepmind code)

# ideas
# --> human training on first 50000 timesteps
# --> reward=(-1) for loss of life


# python libraries
import os, sys
import time
import pickle
import random

import psutil


# open ai gym
import gym
# numpy
import numpy as np
# termcolor
from termcolor import cprint as print_in_color
# python imaging library
from PIL import Image

# scikit-video
# conda install -c conda-forge sk-video
# conda install -c menpo ffmpeg
import skvideo.io

# conda install -c anaconda tensorflow-gpu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Flatten, Dense, Multiply
from tensorflow.keras.optimizers import RMSprop

# ignore deprecation warnings
#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

#np.set_printoptions(threshold=sys.maxsize)

################################################################################

# one epoch corresponds to 50000 weight updates or 30 minutes of training time
# --> implies each epoch is 200,000 timesteps
# --> average reward of 20 at 20 epochs (4M timesteps epsilon=0.05)
# --> average reward of 50 at 30 epochs (6M timesteps epsilon=0.05)
#
# --> time throughput for each timestep
# --> save weights (every million iterations)
# TODO: in paper deepmind set learning rate to 0.00025, but 0.01==>decay in code !??

################################################################################

def return_downsampled_image(numpy_array):
    # 210x160 -> 105x80
    img = Image.fromarray(numpy_array)
    img = img.resize((80, 105))
    downsampled_img = np.asarray(img)
    return downsampled_img

def rgb_to_grayscale(numpy_array):
    r, g, b = numpy_array[:,:,0], numpy_array[:,:,1], numpy_array[:,:,2]
    grayscale_image_array = 0.3 * r + 0.5870 * g + 0.1140 * b
    grayscale_image_array = grayscale_image_array.astype('uint8')
    return grayscale_image_array

def preprocess_frame(frame):
    gray_frame = rgb_to_grayscale(frame)
    downsampled_gray_frame = return_downsampled_image(gray_frame)
    square_downsampled_gray_frame = downsampled_gray_frame[25:,:]
    frame = square_downsampled_gray_frame
    frame = np.reshape(frame, (80,80,1))
    return frame

def preprocess_reward(reward):
    return np.sign(reward)

class generate_game(object):

    def __init__(self):
        # ____________________________________________________________________ #
        self.env = gym.make('BreakoutDeterministic-v4')
        # print(self.env.unwrapped.get_action_meanings())
        # ____________________________________________________________________ #
        self.experience_replay_memory = generate_experience_replay_memory()
        # ____________________________________________________________________ #
        self.current_state = self.start_game()
        # ____________________________________________________________________ #
        self.N_games_completed = 0
        self.average_reward = 0
        # ____________________________________________________________________ #

    def start_game(self):
        self.current_game_total_reward = 0
        # __
        start_frame = self.env.reset()
        NOOP = 0
        frame1, _, _, _ = self.env.step(NOOP)
        frame2, _, _, _ = self.env.step(NOOP)
        frame3, _, _, _ = self.env.step(NOOP)
        frame4, _, _, info = self.env.step(NOOP)
        frame1 = preprocess_frame(frame1)
        frame2 = preprocess_frame(frame2)
        frame3 = preprocess_frame(frame3)
        frame4 = preprocess_frame(frame4)
        # __
        self.N_lives_previous_timestep = int(info['ale.lives'])
        # __
        new_state = np.concatenate((frame1, frame2, frame3, frame4), axis=2)
        return new_state

    def update_average_reward(self):
        current_reward = self.current_game_total_reward
        previous_average_reward = self.average_reward
        w1 = 1.0 / self.N_games_completed
        w2 = (float(self.N_games_completed-1)) / self.N_games_completed
        self.average_reward = w1 * current_reward + w2 * previous_average_reward
        # print_in_color("avg={0} current={1}".format(self.average_reward, current_reward), "cyan")

    def return_next_state(self, next_frame):
        # next state: most recent 3 frames of the previous_state + next frame
        first_three_frames = self.current_state[:,:,1:]
        next_state = np.append(first_three_frames, next_frame, axis=2)
        return next_state

    def next_action(self, action):
        # ____________________________________________________________________ #
        next_frame, reward, bool_game_complete, info = self.env.step(action)
        # __
        N_lives = int(info['ale.lives'])
        bool_lost_a_life = False
        if N_lives != self.N_lives_previous_timestep:
            bool_lost_a_life = True
            self.N_lives_previous_timestep = N_lives
        # __
        next_frame = preprocess_frame(next_frame)
        reward     = preprocess_reward(reward)
        self.current_game_total_reward += reward
        current_state = self.current_state
        next_state    = self.return_next_state(next_frame)
        # __
        if bool_lost_a_life == True:
            experience = (current_state, action, -1, True, next_state)
        else:
            experience = (current_state, action, reward, bool_game_complete, next_state)
        self.experience_replay_memory.add_experience(experience)
        # ____________________________________________________________________ #
        if bool_game_complete == True:
            self.N_games_completed += 1
            self.update_average_reward()
            self.current_state = self.start_game()
        else:
            self.current_state = next_state
        # ____________________________________________________________________ #

    def random_action(self):
        random_action = self.env.action_space.sample()
        #print("random_action={0}",format(random_action))
        self.next_action(random_action)

    def agent_action(self, model):
        agent_action = return_predicted_optimal_action(model, self.current_state)
        #print("agent_action={0}",format(agent_action))
        self.next_action(agent_action)

def init_memory_100000_experiences_and_save_to_disc(game, init_memory_filename):
    #os.remove(init_memory_filename)
    for i in range(0,100000):
        if i % 10000 == 0:
            print_in_color("i={0}".format(i), "cyan")
        game.random_action()
    FILE = open(init_memory_filename, 'wb')
    pickle.dump(game.experience_replay_memory.data, FILE)
    FILE.close()

# linearly anneal epsilon from 1.0 to 0.1 (over course time_step 1-1,000,000)
def return_epsilon_for_given_time_step(time_step):
    if time_step < 100000:
        epsilon = 1.0 - 0.9 * (time_step * 0.00001)
    elif time_step < 200000:
        epsilon = 0.1 - 0.05 * ((time_step % 100000) * 0.00001)
    else:
        epsilon = 0.05
    return epsilon

#
# array of (state, action, reward, bool_terminal_state, next_state) tuples
# --> each state is 80x80x4 array of 4 grayscale frames
#
class generate_experience_replay_memory(object):
    def __init__(self):
        self.data = []
        self.max_length = 200000

    def load_memory(self, filename):
        print("loading experience-replay-memory from disc...")
        with open(filename,'rb') as FILE:
            self.data = pickle.load(FILE)
        print("LEN(DATA)={0}".format(len(self.data)))
        print("DONE loading experience-replay-memory from disc")

    def save_gameplay_mp4(self, filename):
        xp = self.data[-600:]
        list_of_frames = []
        index = 0
        while len(list_of_frames) < 600:
            (current_state, action, reward, bool_terminal_state, next_state) = xp[index]
            list_of_frames.append(current_state[:,:,3])
            index += 1

        array_of_frames = np.asarray(list_of_frames)
        # print_numpy_array(array_of_frames, "array_of_frames")
        skvideo.io.vwrite(filename, array_of_frames)

    def add_experience(self, experience):
        if len(self.data) >= self.max_length:
            #new_data = self.data[10000:]
            #del self.data
            #self.data = new_data
            #del new_data
            self.data = self.data[10000:]
        self.data.append(experience)

################################################################################


def return_DQN_model(N_actions):
    # _____
    # input: (80,80,4)
    # _____
    # conv1: (80,80,4)  ==> Conv(8,8,32)stride4 ==> (20,20,32)
    # conv2: (20,20,32) ==> Conv(4,4,64)stride2 ==> (9,9,64)
    # conv3: (9,9,64)   ==> Conv(3,3,64)stride1 ==> (7,7,64)
    # flat3: (7,7,64)   ==> Flat()              ==> (3136)
    # dens4: (3136)     ==> Dense(512)          ==> (512)
    # dens5: (512)      ==> Dense(N_actions)    ==> (N_Actions)
    # _____
    # output: N_actions
    # ==> output is filtered => targets&outputs both set to 0 for actions not chosen during training update
    # _____
    # N_weights = (64*32)+(16*64)+(9*64)+(3136*512)+(512*4) = 1,611,328

    # __
    input_frames     = Input((80,80,4))
    actions_to_value = Input((4,))
    # __
    normalize_input_frames = Lambda(lambda x: x / 255.0)
    # __
    conv1 = Conv2D(32, (8,8), strides=4, activation='relu', input_shape=(80,80,4))
    conv2 = Conv2D(64, (4,4), strides=2, activation='relu')
    conv3 = Conv2D(64, (3,3), strides=1, activation='relu')
    flatten = Flatten()
    # __
    dense_512 = Dense(512)
    value_actions = Dense(N_actions)
    # __
    select_actions_to_value = Multiply()
    # __

    # ____________________________________________________________________ #
    input  = normalize_input_frames(input_frames)
    # __
    layer1 = conv1(input)
    layer2 = conv2(layer1)
    layer3 = conv3(layer2)
    layer3_flat = flatten(layer3)
    # __
    layer4 = dense_512(layer3_flat)
    output = value_actions(layer4)
    output = select_actions_to_value([output, actions_to_value])
    # ____________________________________________________________________ #
    model = Model(inputs=[input_frames, actions_to_value], outputs=output)
    # ____________________________________________________________________ #
    # RMSprop: divide the gradient by running_average_of_recent_gradients_absolute_magnitude
    # average_gradient_squared = (rho)*average_gradient_squared + (1-rho)*gradient²
    # average_gradient = sqrt(average_gradient_squared)+epsilon
    # gradient /= average_gradient
    # parameters from 'Human level control through Deep Reinforcement Learning'
    deepmind_RMSprop = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    # ____________________________________________________________________ #
    # average(error²)
    # --> error = minibatch targets - DQN_output
    # --> minibatch targets for game with 4 moves will be [0, t, 0, 0]
    # --> where t = r + max_action(Q[s,a,theta])
    model.compile(loss='mean_squared_error', optimizer=deepmind_RMSprop)
    # ____________________________________________________________________ #
    return model


def return_predicted_optimal_action(model, current_state):
    current_state = np.reshape(current_state, (1,80,80,4))
    actions_to_value = np.array([1,1,1,1])
    actions_to_value = actions_to_value.reshape(1,4)
    current_state_predicted_action_values  = model.predict([current_state, actions_to_value])
    current_state_predicted_optimal_action = np.argmax(current_state_predicted_action_values)
    #print("current_state_predicted_action_values", current_state_predicted_action_values)
    #print("max_action[current_state_predicted_action_values]", current_state_predicted_optimal_action)

    return int(current_state_predicted_optimal_action)


def return_bool_next_move_is_random(timestep):
    epsilon = return_epsilon_for_given_time_step(timestep)
    if random.random() < epsilon:
        return True
    else:
        return False

def train_model():
    # __
    game = generate_game()
    init_memory_filename = "init_memory_100000_experiences.pickle"
    # _
    init_memory_100000_experiences_and_save_to_disc(game, init_memory_filename)
    game.experience_replay_memory.load_memory(init_memory_filename)
    # __
    N_actions = 4
    DQN_model = return_DQN_model(N_actions)
    # __

    t0 = time.time()
    process = psutil.Process(os.getpid())

    for training_iteration_i in range(0, 10000):
        t1 = time.time()
        # __
        if training_iteration_i % 100 == True:
            model_filename = "replay/2018_08_30/DQN.h5"
            gameplay_filename = "replay/2018_08_30/GAMEPLAY.mp4"
            try:
                os.remove(model_filename)
                os.remove(gameplay_filename)
            except:
                pass
            DQN_model.save(model_filename)
            game.experience_replay_memory.save_gameplay_mp4(gameplay_filename)
        # __
        ########################################################################################################
        ########################################################################################################
        '''_______________________________________________________________'''
        print_in_color("memory useX={0}".format(process.memory_info()[0]), "red")  # in bytes

        N_samples = 8000
        list_of_next_states = []
        list_of_previous_experiences = []
        for i in range(0, N_samples):
            experience = (state, action, reward, bool_game_complete, next_state) = random.choice(game.experience_replay_memory.data)
            list_of_previous_experiences.append(experience)
            list_of_next_states.append(next_state)
        ######################next_state_values = value_next_states(previous_model, list_of_next_states, N_samples)
        #############################
        #############################



        next_state_actions_to_value = np.array([1.0, 1.0, 1.0, 1.0])
        # 4 -> 1x4
        next_state_actions_to_value = next_state_actions_to_value.reshape(1,4)
        # 1x4 -> 800x4
        next_state_actions_to_value = np.repeat(next_state_actions_to_value, N_samples, axis=0)
        next_states = np.reshape(list_of_next_states, (N_samples,80,80,4))

        next_state_predicted_action_values = DQN_model.predict([next_states, next_state_actions_to_value], batch_size=256)
        next_state_predicted_values = np.max(next_state_predicted_action_values, axis=1)





        #############################
        #############################
        # __
        list_of_states  = []
        list_of_actions = []
        list_of_targets = []
        # __
        for i in range(0, N_samples):
            #if i%1000==0:
            #    print_in_color(i, "red")
            #    print(time.time()-t0)
            #    t0=time.time()
            (state, action, reward, bool_game_complete, next_state) = list_of_previous_experiences[i]
            # __
            list_of_states.append(state)
            select_action = [0.0, 0.0, 0.0, 0.0]
            select_action[action] = 1.0
            select_action = np.asarray(select_action)
            list_of_actions.append(select_action)
            # __
            if bool_game_complete == True:
                expected_reward = reward
            else:
                gamma = 0.99
                expected_reward = reward + gamma * next_state_predicted_values[i]
            target = [0.0, 0.0, 0.0, 0.0]
            target[action] = expected_reward
            target = np.asarray(target)
            list_of_targets.append(target)
            # __


        list_of_states  = np.reshape(list_of_states, (N_samples,80,80,4))
        list_of_actions = np.reshape(list_of_actions, (N_samples, 4))
        training_inputs  = [list_of_states, list_of_actions]
        training_targets = np.reshape(list_of_targets, (N_samples,4))
        ########################################################################################################
        ########################################################################################################
        print("len(previous_experiences)={0}".format(len(game.experience_replay_memory.data)))
        #previous_experiences = None
        # __
        ta= time.time()
        ####print_in_color("memory use1={0}".format(process.memory_info()[0]), "red")  # in bytes
        #print_in_color("memory useA={0}".format(process.memory_info()[0]), "cyan")  # in bytes
        DQN_model.fit(training_inputs, training_targets, batch_size=32)
        #print_in_color("memory useB={0}".format(process.memory_info()[0]), "cyan")  # in bytes

        ####print_in_color("memory use2={0}".format(process.memory_info()[0]), "red")  # in bytes

        del list_of_next_states
        del list_of_previous_experiences

        del next_state_actions_to_value
        del next_states
        del next_state_predicted_action_values
        del next_state_predicted_values

        del list_of_states
        del list_of_actions
        del list_of_targets

        del training_inputs
        del training_targets

        '''__________________________________________________________________'''

        tb = time.time()
        #print_in_color("memory useY={0}".format(process.memory_info()[0]), "red")  # in bytes

        # __
        t2 = time.time()
        N_agent_actions = 0
        for gameplay_step_i in range(0, 1000):
            timestep = (training_iteration_i * 1000) + gameplay_step_i
            bool_next_move_is_random = return_bool_next_move_is_random(timestep)
            if bool_next_move_is_random == True:
                game.random_action()
            else:
                N_agent_actions += 1
                game.agent_action(DQN_model)
        t3 = time.time()

        #print_in_color("memory useZ={0}".format(process.memory_info()[0]), "red")  # in bytes


        print("Iteration={0} N_agent_actions={1}".format(training_iteration_i, N_agent_actions))
        print_in_color("AVERAGE SCORE = {0:.2f}".format(game.average_reward), 'blue')
        print_in_color("TRAINING TIME = {0:.2f}".format(t2-t1), "cyan")
        print_in_color("GAMEPLAY TIME = {0:.2f}".format(t3-t2), "red")

        #xp_checkpoint_filename = "replay/2018_08_30/xp.pickle"
        #FILE = open(xp_checkpoint_filename, 'wb')
        #pickle.dump(game.experience_replay_memory.data, FILE)
        #FILE.close()



if __name__ == "__main__":
    train_model()
