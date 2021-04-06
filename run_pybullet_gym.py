import numpy as np
import gym
import pybulletgym
import imageio
import pickle
import os
from data_buffer import ReplayBuffer

import argparse
def argsparser():
    parser = argparse.ArgumentParser("Implementation")
    parser.add_argument('--eval_mode', help='evaluation mode, human or rgb_array', default="human", type=str)
    parser.add_argument('--eval_random', help='evaluate -> randomly get actions or not', default=1, type=int)
    parser.add_argument('--e_greedy_evaluation', help='probability of random action during evaluation', default=0.05, type=int)
    parser.add_argument('--lr', help='learning rate', default=3e-4)
    parser.add_argument('--l2', help='lambda', default=1e-3)
    parser.add_argument('--batch_size', help='size of minibatch for minibatch-SGD', default=64)
    parser.add_argument("--gamma", default=0.99, type=float)

    parser.add_argument('--env_id', help='choose the gym env', default='HopperPyBulletEnv-v0')
    parser.add_argument('--seed', help='random seed for repeatability', default=0)
    parser.add_argument('--max_episode_length', help='maximum length for a single run', default=1000)
    parser.add_argument("--max_timesteps", default=1e8, type=float)
    parser.add_argument("--eval_freq", default=100, type=float)
    parser.add_argument("--checkpoint_freq", default=2e5, type=float)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='logs')
    parser.add_argument('--gif_dir', help='the directory to save GIFs file', default='gifs')
    parser.add_argument('--data_dir', help='the directory to save model', default='hopper_data')
    parser.add_argument('--custom_id', help='user defined model name', default='default')

    return parser.parse_args()


def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    imageio.mimsave(f'{path}{"mujuco_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1 / 30)

class policy:
    def __init__(self, args, action_space, obs_space):
        self.args = args
        self.action_space = action_space
        self.obs_space = obs_space
        #Implement here
        print("TBD")

    def get_random_action(self, batch_size):
        #for debugging or e-greedy
        return np.tanh(np.random.normal(0, 1, size=(batch_size, self.action_space.shape[0])))

    def get_action(self, ob):
        #Implement, choose what arguments are needed
        print("TBD")

    def train_step(self, buffer):
        obs, obs_next, acs, rew, terminal = buffer.sample(self.args.batch_size)
        #implement here ... 
        print("TBD")

def evaluate(args, policy_network, env, train_step, e_greedy_evaluation=0, min_gap_between_random=5, random_sampling=True, mode="rgb_array"): #Note the mode can be rgb_array -> without gui, or human -> with gui
    max_eps_len = args.max_episode_length
    frames_list = []

    eps_reward = 0
    eps_len = 0
    if mode == "human":
        env.render(mode="human")
    current_state = env.reset()
    current_gap = 0
    for i in range(max_eps_len):
        current_state = np.expand_dims(current_state, axis=0)
        if random_sampling or (np.random.uniform(0, 1) < e_greedy_evaluation and current_gap >= min_gap_between_random):
            action = policy_network.get_random_action(1)
            current_gap = 0
        else:
            action = policy_network.get_action(current_state)
            current_gap += 1

        action = action[0]
        current_state, reward, terminal, _ = env.step(action)
        eps_reward += reward
        eps_len += 1

        rendered_frame = env.render(mode="rgb_array")
        frames_list.append(rendered_frame)

        if terminal:
            break;

    generate_gif(train_step, frames_list, int(np.round(eps_reward)), args.gif_dir + "/gif_e_greedy_" + str(e_greedy_evaluation) + "_")
    return eps_reward, eps_len

def load_data(args, buffer):
    data_files = os.listdir(args.data_dir + "/")
    for data_file in data_files:
        raw_data = open(args.data_dir + "/" + data_file, "rb")
        data = pickle.load(raw_data)
        raw_data.close()

        num_samples_per_trajectory = len(data["obs"])
        for i in range(num_samples_per_trajectory):
            #Note, i accidently saved the terminal as acs and actions as term
            #Furthermore, the action range is [-1, 1] so we fix that
            buffer.add(np.squeeze(data["obs"][i]), np.squeeze(data["obs"][min(i+1, num_samples_per_trajectory - 1)]),
                       np.clip(np.squeeze(data["term"][i]), -1, 1), np.squeeze(data["rew"][i]), np.squeeze(data["acs"][i]))

def train():
    args = argsparser()
    #set seed
    np.random.seed(args.seed)

    #make env
    env = gym.make(args.env_id)
    ob_space = env.observation_space
    ac_space = env.action_space
    buffer = ReplayBuffer(100000)
    load_data(args, buffer)
    #build network
    policy_network = policy(args, ac_space, ob_space)
    train_step_since_eval = 0
    train_step_since_checkpoint = 0


    #set the directories
    args.log_dir = "./" + args.log_dir + "/" + args.env_id  + "_seed_" + str(args.seed) + "_" + args.custom_id
    args.gif_dir = "./" + args.gif_dir + "/" + args.env_id  + "_seed_" + str(args.seed) + "_" + args.custom_id
    args.checkpoint_dir = "./" + args.checkpoint_dir + "/" + args.env_id  + "_seed_" + str(args.seed) + "_" + args.custom_id

    #make them if they do not already exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    #train loop
    for train_step in range(int(args.max_timesteps)):
        #implement training loop here 
        #end
        train_step_since_eval += 1
        train_step_since_checkpoint += 1
        if train_step_since_eval > args.eval_freq:
            train_step_since_eval = 0
            # Note random sampling should be set to false when you start actually training
            eval_rew_list = []
            eval_len_list = []
            eval_e_greedy_rew_list = []
            eval_e_greedy_len_list = []
            for i in range(5):
                eval_rew, eval_len = evaluate(args, policy_network, env, train_step, random_sampling=args.eval_random,
                                            mode=args.eval_mode, e_greedy_evaluation=0)

                eval_rew_list.append(eval_rew)
                eval_len_list.append(eval_len)
                eval_rew, eval_len = evaluate(args, policy_network, env, train_step, random_sampling=args.eval_random,
                                            mode=args.eval_mode, e_greedy_evaluation=args.e_greedy_evaluation)
                eval_e_greedy_rew_list.append(eval_rew)
                eval_e_greedy_len_list.append(eval_len)
            print("-------------------------------------------")
            print("No E-greedy eval rew:", np.mean(eval_rew_list),"+-", np.std(eval_rew_list),  "eval len: ", np.mean(eval_len_list))
            print("E-greedy eval rew:", np.mean(eval_e_greedy_rew_list),"+-", np.std(eval_e_greedy_rew_list), "eval len: ", np.mean(eval_e_greedy_len_list))
            print("-------------------------------------------")

        if train_step_since_checkpoint > args.eval_freq:
            train_step_since_checkpoint = 0
            #save model here
            print("TBD")

if __name__ == "__main__":
    train()