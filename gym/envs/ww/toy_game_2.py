import sys
import math
import string

import cv2
import numpy as np
import scipy
from scipy import ndimage

import gym
from gym import spaces, utils
from gym.utils import seeding

class ToyDiscreteEnv3(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 5}

    def __init__(self, board_size=50, num_classes=1, patch_size=5, rf_size=5, context_size=5, simple=False, 
                    obj_mode='square', saliency_mode='simple', downsample_mode='avg_pooling'):

        self.board_size = board_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.rf_size = rf_size
        self.context_size = context_size
        self.simple = simple

        self.obj_mode = obj_mode 
        self.saliency_mode = saliency_mode 
        self.downsample_mode = downsample_mode

        self.I = np.zeros([board_size, board_size, num_classes + 1])
        self.S = np.zeros([board_size, board_size, 1])

        self.prev_c = None
        self.prev_pos = None
        self.target_class = None
        self.l = None
        self.counter = np.zeros(self.num_classes, dtype='int32') 
        self.picked = {} # 1: picked, 0: NOT picked

        self.action_space = spaces.Box(low=0, high=1, shape=(2, )) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(rf_size, rf_size, self.num_classes+2))

        self._seed() 
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        assert self.obj_mode in ['square'], "Unsupported form: %s" % self.obj_mode

        self.prev_c = None
        self.prev_pos = None
        self.target_class = self.np_random.choice(self.num_classes)
        
        self.I *= 0
        self.I[:, :, 0] += 1
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
            p = patches[i]
            x_patch = p // patch_num
            y_patch = p % patch_num
            xstart = x_patch * patch_size 
            ystart = y_patch * patch_size
            xchoices = patch_size if x_patch != patch_num - 1 else self.board_size % patch_size + patch_size
            ychoices = patch_size if y_patch != patch_num - 1 else self.board_size % patch_size + patch_size
            self._put_pieces(xchoices, ychoices, xstart, ystart, i)

        self.S = self._get_saliency() 
        self.S_0 = self._downsample()
        self.l = np.argmax(self.S_0) 

        return self._get_return_board() 

    def _put_pieces(self, xchoices, ychoices, xstart, ystart, c):
        if self.obj_mode == 'square':
            assert xchoices == ychoices, "Incorrect shape for 'square' format"
            block = max(min(xchoices // 3, 1), (xchoices - 1) // 2)
            self.num_pieces = (block**2)*4
            x = np.hstack([np.arange(xchoices)[:block], np.arange(xchoices)[-block:]]) + xstart
            x = np.repeat(x, self.num_pieces // 4)
            y = np.hstack([np.arange(ychoices)[:block], np.arange(ychoices)[-block:]]) + ystart
            y = np.tile(y, self.num_pieces // 4)
            self.I[x, y, :] = 0
            self.I[x, y, c + 1] = 1
            for i in zip(x, y):
                self.picked[i] = 0

    def _get_saliency(self):
        assert self.saliency_mode in ['blur', 'simple'], "Unsupported saliency mode %s" % self.saliency_mode
        S = 1.0 - np.array(self.I[:, :, 0] == 1, dtype='float64') 
        if self.saliency_mode == 'blur':
            blurred_S = scipy.ndimage.gaussian_filter(S, sigma=2)
            return np.expand_dims(blurred_S, axis=2)
        elif self.saliency_mode == 'simple':
            return np.expand_dims(S, axis=2) 


    def _downsample(self):
        assert self.downsample_mode in ['avg_pooling'], "Unsupported downsample mode %s" % self.downsample_mode
        # downsample strategy: avg_pooling, ksize=[rf_size, rf_size], stride=[rf_size, rf_size] 
        if self.downsample_mode == 'avg_pooling':
            assert self.board_size % self.rf_size == 0
            new_size = self.board_size // self.rf_size
            new_board = np.zeros([new_size, new_size])

            for i in range(new_size):
                for j in range(new_size):
                    new_board[i, j] = np.mean(self.S[i*self.rf_size:(i+1)*self.rf_size, j*self.rf_size:(j+1)*self.rf_size])

            return new_board


    def _get_return_board(self):
        # get S@l_t, I@l_t, S_0@l_t
        if self.downsample_mode == 'avg_pooling':
            x = self.l // (self.board_size // self.rf_size)
            y = self.l % (self.board_size // self.rf_size)
            self.s_l = (self.S[x*self.rf_size:(x+1)*self.rf_size, y*self.rf_size:(y+1)*self.rf_size]).copy()
            self.i_l = (self.I[x*self.rf_size:(x+1)*self.rf_size, y*self.rf_size:(y+1)*self.rf_size][:, :, 1:]).copy()
 
            pad_size = self.context_size // 2 if self.context_size % 2 == 0 else self.context_size // 2 + 1
            xstart = x - self.context_size // 2 + pad_size
            xend = x + pad_size + pad_size
            ystart = y - self.context_size // 2 + pad_size
            yend = y + pad_size + pad_size
            self.s0_l = np.lib.pad(self.S_0, pad_size, mode='constant', constant_values=0)[xstart:xend, ystart:yend]
            self.s0_l = np.expand_dims(self.s0_l, axis=2)

            return np.concatenate([self.i_l, self.s_l, self.s0_l], axis=2).squeeze()
        

    def _step(self, action):
        assert self.action_space.contains(action), "no such action"

        reward = 0
        done = False

        x, y = self._get_loc(action)

        reward -= self._dist((x, y)) 
        self.prev_pos = (x, y)

        if self.simple:
            x_low = x - 1 if x >= 1 else 0
            x_high = x + 2 if x <= self.board_size - 2 else self.board_size
            y_low = y - 1 if y >= 1 else 0
            y_high = y + 2 if y <= self.board_size - 2 else self.board_size
            glimpse = [np.sum([self.board[x_low:x_high, y_low:y_high, i] == 1]) for i in range(self.num_classes + 1)]
            # most of the remaining pieces is included in the box
            max_class = int(np.max(np.argwhere(glimpse[1:]==np.max(glimpse[1:]))))
            cond1 = (np.max(glimpse[1:]) >= (self.num_pieces - self.counter[max_class]) // 2) 
            # one and only one class is included in the bounding box
            cond2 = np.sum(glimpse[1:]) == np.max(glimpse[1:]) != 0 
            # one of the class is majority 
            cond3 = (np.max(glimpse[1:]) > np.sum(glimpse[1:]) / 3 * 2) 
            
            if (np.max(glimpse) >= (x_high - x_low) * (y_high - y_low) // 2 + 1) and np.argmax(glimpse) != 0:
                c = np.argmax(glimpse)
            elif (cond1 and cond2) or cond3:
                c = np.argmax(glimpse[1:]) + 1
            else:
                c = 0
        else:
            c = int(np.where(self.I[x, y, :] == 1)[0])

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
                    
            else:
                self.board[x, y, c] = 0
                self.board[x, y, 0] = 1
                self.counter[int(c-1)] += 1
                self.picked[(x, y)] = 1

                
            done = False if (c == self.prev_c if self.prev_c is not None else True) else True
            if self.counter[int(c-1)] == self.num_pieces: # if the agent has seen every piece in this class
                reward += 10
                self.prev_c = None
            else:
                self.prev_c = c
        
        if (self.counter[self.target_class] == self.num_pieces):
            done = True
            # reward += 1000

        config = [] 
        return (self._get_return_board(), reward, done, config) 

    '''
    def _get_loc(self, action):
        size = self.rf_size * self.context_size
        rel_idx = action * size
        x = 

        return  
    '''
    def _dist(self, pos):
        if self.prev_pos is None:
            return 0
        else:
            return np.sum(abs(np.array(self.prev_pos) - np.array(pos)))

    def _render(self, mode='human', close=False):
        if close:
            return
        board = self.I
        if mode == 'human':
            outfile = sys.stdout
            outfile.write(self._repr(board))
            return outfile
        elif mode == 'rgb_array':
            img = self._to_img(board) 
            return img


    def _to_img(self, board):
        board *= 50 
        if self.prev_pos is not None:
            board[self.prev_pos] = 500
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
                if (i, j) != self.prev_pos:
                    out += utils.colorize(class_list[int(np.argmax(board[i, j]))], self.class_color[int(np.argmax(board[i, j]))])
                else:
                    out += utils.colorize(class_list[int(np.argmax(board[i, j]))], "red", highlight=True)
            out += '|\n'
        out += '   +'
        out += '---'*board_size
        out += '+\n'
        return out
