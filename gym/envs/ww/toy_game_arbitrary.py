import sys
import math
import string

import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

import gym
from gym import spaces, utils
from gym.utils import seeding


def softmax(x, alpha):
    exp = np.exp(x / alpha)
    return exp / np.sum(exp)

class ToyContinuousEnv1(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 5}

    def __init__(self, board_size=20, num_classes=2, max_pieces=9, mean_pieces=9, min_pieces=2, patch_size=6, 
                    k=4, discrete=True):
        
        self.board_size = board_size
        self.num_classes = num_classes
        self.max_pieces = max_pieces # each class contains [0, max_pieces] pieces
        self.mean_pieces = mean_pieces
        self.min_pieces = min_pieces
        self.patch_size = patch_size
        self.discrete = discrete

        self.k = k if k is not None else self.max_pieces*self.num_classes 

        self.n = board_size * board_size
        self.board = np.zeros((board_size, board_size, num_classes+1))
        self.board[:, :, 0] = 1

        self.prev_c = None
        self.pos = None
        self.first_act = None
        self.counter = np.zeros(self.num_classes, dtype='int32') 
        self.picked = {} # 1: picked, 0: NOT picked

        self.action_space = spaces.Discrete(self.k) if discrete else spaces.Box(low=0, high=1, shape=(2,)) 
        self.observation_space = spaces.Box(low=0, high=1, shape=((self.k + 1) * (self.num_classes + 2)))

        self._seed() 
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.prev_c = None
        self.pos = None
        # simulating pi_0: choose a target class
        # TODO may need to remove the class information
        self.target_class = self.np_random.choice(self.num_classes)
        self.num_pieces = np.ones(self.num_classes, dtype='int8') * self.min_pieces
        self.num_pieces[self.target_class] = max(min(int(3 * self.np_random.randn()) + self.mean_pieces, self.max_pieces), self.min_pieces)
        # self.num_pieces[self.target_class] = self.max_pieces  

        self.board *= 0
        self.board[:, :, 0] += 1
        self.counter *= 0

        colors = ['gray', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'crimson']
        self.class_color = self.np_random.choice(colors[1:], self.num_classes+1)
        self.class_color[0] = 'gray' 

        # divide the board into several patches
        # choose num_classes patches to generate
        if self.patch_size is None:
            patch_size = max(self.board_size // (int(math.ceil(math.sqrt(self.num_classes))) + 1), int(math.ceil(math.sqrt(self.max_pieces))))
        else:
            patch_size = self.patch_size
        patch_num = self.board_size // patch_size
        patches = self.np_random.choice(patch_num**2, self.num_classes, replace=False)
        for i in range(self.num_classes):
            num_piece = self.num_pieces[i]
            p = patches[i]
            x_patch = p // patch_num
            y_patch = p % patch_num
            xstart = x_patch * patch_size 
            ystart = y_patch * patch_size
            xchoices = patch_size 
            ychoices = patch_size
            
            if self.board_size >= 9 and (patch_size-1)**2 >= self.max_pieces:
                xstart += 1
                ystart += 1
                xchoices -= 1
                ychoices -= 1
            
            num_piece = self._put_pieces(xchoices, ychoices, num_piece, xstart, ystart, i)
            self.num_pieces[i] = num_piece 
            num_overlaps = num_piece - np.sum(self.board[:,:,i+1] == 1)
            while num_overlaps > 0:
                self._put_pieces(xchoices, ychoices, num_overlaps, xstart, ystart, i)
                num_overlaps = num_piece - np.sum(self.board[:,:,i+1] == 1)

        self.first_act = np.where(self.board.reshape([-1, self.num_classes+1])[:, self.target_class+1] == 1)[0][0]

        return self._get_top_k()

    def _put_pieces(self, xchoices, ychoices, num_piece, xstart, ystart, c):
        x = self.np_random.choice(xchoices, num_piece) + xstart 
        y = self.np_random.choice(ychoices, num_piece) + ystart
        self.board[x, y, :] = 0
        self.board[x, y, c + 1] = 1
        for i in zip(x, y):
            self.picked[i] = 0

        return num_piece

    def _get_top_k(self):
        sal = np.array(self.board[:, :, 0] == 0, dtype='float64')
        size = self.board_size**2
        top_k = np.random.choice(np.arange(size), p=np.ravel(softmax(sal, 0.1)), size=self.k, replace=True)
        self.top_k = self.board.copy().reshape([-1, self.board.shape[-1]])[top_k]
        self.top_k = np.hstack([np.zeros([self.k, 1]), self.top_k])
        self.top_k[:, 0] = top_k // self.board_size
        self.top_k[:, 1] = top_k % self.board_size
        last_act = np.zeros(self.num_classes + 2)
        last_act[0] = self.pos[0] if self.pos is not None else self.first_act // self.board_size 
        last_act[1] = self.pos[1] if self.pos is not None else self.first_act % self.board_size
        last_act[self.target_class+2] = 1
        self.top_k = np.vstack([self.top_k, last_act])
        if self.discrete:
            self.obs = np.insert(self.top_k, np.arange(self.k)[1:], self.top_k[-1], axis=0)
            return np.expand_dims(self.obs, axis=2)
        else:
            self.obs = self.top_k.copy()
            self.obs[:, :2] /= self.board_size
            return self.obs.ravel()

    def _step(self, a):
        assert self.action_space.contains(a), "no actions"
        reward = 0
        done = False
        
        if (self.discrete and a == self.k):
            return (self._get_top_k(), -1, False, [])
        
        if self.discrete:
            x = int(self.top_k[a][0])
            y = int(self.top_k[a][1])
        else:
            x = min(int(a[0]*self.board_size), self.board_size-1)
            y = min(int(a[1]*self.board_size), self.board_size-1)

        reward -= 0.2 * self._dist((x, y)) 
        self.pos = (x, y)

        c = int(np.where(self.board[x, y, :] == 1)[0])

        if c == 0:      # if the agent sees nothing
            reward += -1 # it will receive a -1 reward
            try:
                if self.picked[(x, y)] == 1:  # if the agent looks at a previously picked area
                    reward -= 10             # it will receive a -10 reward
            except KeyError:
                pass
        else:
            self.board[x, y, c] = 0
            self.board[x, y, 0] = 1
            self.counter[int(c-1)] += 1
            self.picked[(x, y)] = 1

            reward += -50 if c != self.target_class + 1 else 1 

            if self.counter[int(c-1)] == self.num_pieces[int(c-1)]: # if the agent has seen every piece in this class
                self.prev_c = None
            else:
                self.prev_c = c
        
        if (self.counter[self.target_class] == self.num_pieces[self.target_class]):
            done = True
            reward += 100 
        
        # if ((not self.discrete) and np.sum(a) <= 0.01 / self.board_size):
            # done = True
            # if (self.counter[self.target_class] == self.num_pieces[self.target_class]):
                # reward -= 100
        
        return (self._get_top_k(), reward, done, [])

    def _dist(self, pos):
        if self.pos is None:
            return 0
        else:
            return np.sum(abs(np.array(self.pos) - np.array(pos)))

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            outfile = sys.stdout
            outfile.write(self._repr(self.board))
            return outfile
        elif mode == 'rgb_array':
            img = self._to_img() 
            return img


    def _to_img(self):
        board = np.argmax(self.board, axis=2)*50
        if self.pos is not None:
            board[self.pos] = 500
        cmap = plt.get_cmap('CMRmap')
        rgb_img = np.delete(cmap(board), 3,2)
        img = cv2.resize(rgb_img, (210, 210), interpolation=0)
        img =  np.array(img*500, dtype='uint8')
        return img

    def _repr(self, board):
        board_size = board.shape[0]
        class_list = [' . '] + [ ' '+s+' ' for s in string.ascii_lowercase ]
        out = '\n   +'
        out += '---'*board_size
        out += '+\n'
        for i in range(board_size):
            out += '   |'
            for j in range(board_size):
                if (i, j) != self.pos:
                    out += utils.colorize(class_list[int(np.argmax(board[i, j]))], self.class_color[int(np.argmax(board[i, j]))])
                else:
                    out += utils.colorize(class_list[int(np.argmax(board[i, j]))], "red", highlight=True)
            out += '|\n'
        out += '   +'
        out += '---'*board_size
        out += '+\n'
        return out
