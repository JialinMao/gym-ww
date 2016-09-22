import sys
import cv2
import matplotlib.pyplot as plt
import math
import string
import numpy as np

import gym
from gym import spaces, utils
from gym.utils import seeding

class ToyDiscreteEnv2(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 5}

    def __init__(self, board_size, num_classes, max_pieces, mean_pieces, simple, patch_size=None, picked_info=False):
        self.board_size = board_size
        self.num_classes = num_classes
        self.max_pieces = max_pieces # each class contains [0, max_pieces] pieces
        self.mean_pieces = mean_pieces
        self.patch_size = patch_size
        self.simple = simple
        self.picked_info = picked_info

        self.n = board_size * board_size
        self.board = np.zeros((board_size, board_size, num_classes+1))
        self.board[:, :, 0] = 1

        self.prev = None
        self.pos = None
        self.target_class = None
        self.first_act = None
        self.counter = np.zeros(self.num_classes, dtype='int32') 
        self.picked = {} # 1: picked, 0: NOT picked

        self.action_space = spaces.Discrete(self.n) 
        depth = 3 if self.picked_info else 2
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_classes, num_classes, depth))

        self._seed() 
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.num_pieces = (3 * self.np_random.randn(self.num_classes)).astype('int8') + self.mean_pieces
        self.num_pieces[self.num_pieces > self.max_pieces] = self.max_pieces
        self.num_pieces[self.num_pieces < 2] = 2 
        self.prev = None
        self.pos = None
        self.target_class = self.np_random.choice(self.num_classes)
        
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
            xchoices = patch_size if x_patch != patch_num - 1 else self.board_size % patch_size + patch_size
            ychoices = patch_size if y_patch != patch_num - 1 else self.board_size % patch_size + patch_size
            ''' 
            if self.board_size >= 9:
                xstart += 1
                ystart += 1
                xchoices -= 1
                ychoices -= 1
            '''
            x, y = self._put_pieces(xchoices, ychoices, num_piece, xstart, ystart, i)
            num_overlaps = num_piece - np.sum(self.board[:,:,i+1] == 1)
            while num_overlaps > 0:
                self._put_pieces(xchoices, ychoices, num_overlaps, xstart, ystart, i)
                num_overlaps = num_piece - np.sum(self.board[:,:,i+1] == 1)

        self.first_act = np.where(self.board.reshape([-1, self.num_classes+1])[:, self.target_class+1] == 1)[0][0]
        self._get_return_board()

        return (self.return_board, np.zeros_like(self.return_board)) 
        # return np.expand_dims(np.sum(self.return_board, axis=2), axis=2)

    def _put_pieces(self, xchoices, ychoices, num_piece, xstart, ystart, c):
        x = self.np_random.choice(xchoices, num_piece) + xstart 
        y = self.np_random.choice(ychoices, num_piece) + ystart
        self.board[x, y, :] = 0
        self.board[x, y, c + 1] = 1
        for i in zip(x, y):
            self.picked[i] = 0
        return zip(x, y)[0]

    def _get_return_board(self):
        self.return_board = np.ones([self.board_size, self.board_size, 2])
        # changed
        self.return_board[:, :, 0] = self.board[:, :, 0]
        self.return_board[:, :, 1] = np.array((self.board[:, :, 0] == 0), dtype='int8')
        self.return_board = np.expand_dims(np.argmax(self.return_board, axis=2), axis=2)
        # The second channel indicates the last action
        '''
        if self.pos is not None:
            x, y = self.pos
            self.return_board[x, y, 1] = 5 
        '''

    def _step(self, action):

        assert self.action_space.contains(action), "no actions"
        reward = 0
        done = False
        
        self._get_return_board()
        last_act = np.zeros_like(self.return_board)
        action += 1
        self.picked_board = np.concatenate((self.return_board, np.zeros([self.board_size, self.board_size, 1])), axis=2) 
        
        x = (action - 1) // self.board_size
        y = (action - 1) % self.board_size 
        last_act[x, y, 0] = 1
        reward -= self._dist((x, y)) 
        self.pos = (x, y)

        if self.simple:
            x_low = x - 1 if x >= 1 else 0
            x_high = x + 2 if x <= self.board_size - 2 else self.board_size
            y_low = y - 1 if y >= 1 else 0
            y_high = y + 2 if y <= self.board_size - 2 else self.board_size
            glimpse = [np.sum([self.board[x_low:x_high, y_low:y_high, i] == 1]) for i in range(self.num_classes + 1)]
            # most of the remaining pieces is included in the box
            max_class = int(np.max(np.argwhere(glimpse[1:]==np.max(glimpse[1:]))))
            cond1 = (np.max(glimpse[1:]) >= (self.num_pieces[max_class] - self.counter[max_class]) // 2) 
            # one and only one class is included in the bounding box
            cond2 = np.sum(glimpse[1:]) == np.max(glimpse[1:]) != 0 
            # one of the class is majority 
            cond3 = (np.max(glimpse[1:]) > np.sum(glimpse[1:]) / 3 * 2) 
            
            if (np.max(glimpse) >= (x_high - x_low) * (y_high - y_low) // 2 + 1) and np.argmax(glimpse) != 0:
                c = np.argmax(glimpse)
            elif (cond1 and cond2) or cond3:
            # elif cond1 and cond2:
                c = np.argmax(glimpse[1:]) + 1
            else:
                c = 0
        else:
            c = int(np.where(self.board[x, y, :] == 1)[0])

        if c == 0:      # if the agent sees nothing
            reward += -1 # it will receive a -1 reward
            try:
                if self.picked[(x, y)] == 1:  # if the agent looks at a previously picked area
                    reward += -10             # it will receive a -10 reward
            except KeyError:
                pass
        else:
            if self.simple:
                self.counter[int(c-1)] += glimpse[c] 
                mask = self.board[x_low:x_high, y_low:y_high, c] == 1
                self.board[x_low:x_high, y_low:y_high][mask] = 0
                self.board[x_low:x_high, y_low:y_high, 0][mask] = 1

                for i in zip(np.tile(np.arange(x_low, x_high), 2), np.repeat(np.arange(y_low, y_high), 2)):
                    self.picked[i] = 1
                    if self.picked_info:
                        self.picked_board[i[0], i[1], :] = 0
                        self.picked_board[i[0], i[1], -1] = 1 

            else:
                self.board[x, y, c] = 0
                self.board[x, y, 0] = 1
                self.counter[int(c-1)] += 1
                self.picked[(x, y)] = 1

                if self.picked_info:
                    self.picked_board[x, y, :] = 0
                    self.picked_board[x, y, -1] = 1 

            done = False if (c == self.prev if self.prev is not None else True) else True
            if self.counter[int(c-1)] == self.num_pieces[int(c-1)]: # if the agent has seen every piece in this class
                reward += 10
                self.prev = None
            else:
                self.prev = c
        
        if (self.counter[self.target_class] == self.num_pieces[self.target_class]):
            done = True
            # lastchange
            # reward += 1000

        # config = [self.prev, c, cond1, cond2, cond3]
        config = [] 
        
        if self.picked_info:
            return (self.picked_board, reward, done, config)

        self._get_return_board()
        return ((self.return_board, last_act), reward, done, config) 
        # return (np.expand_dims(np.sum(self.return_board, axis=2), axis=2), reward, done, config)

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
        elif mode[:-1] == 'downsample':
            new_board = self._downsample(int(mode[-1]))
            outfile = sys.stdout
            outfile.write(self._repr(new_board))
            return outfile

    def _downsample(self, times):
        board = np.argmax(self.board, axis=2)
        for i in range(times):
            board_size = board.shape[0]
            new_size = board_size // 2 
            new_board = np.zeros([new_size, new_size, self.num_classes+1]) if board_size%2==0 else np.zeros([new_size+1, new_size+1, self.num_classes+1])
            for x in range(new_size):
                for y in range(new_size):
                   new_board[x, y, np.max(board[2*x:2*(x+1), 2*y:2*(y+1)])] = 1
            if self.board_size % 2 != 0:
                new_board[-1, :, board[-1, :]] = 1
                new_board[:, -1, board[:, -1]] = 1
            board = np.argmax(new_board, axis=2)
        return new_board


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
            #out += str(i) + '  |'
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
