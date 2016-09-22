from gym.envs.ww.toy_game import ToyDiscreteEnv
# game-v1, several classes, give a 'first choice', aim is to pick all pieces of this class
from gym.envs.ww.toy_game_1 import ToyDiscreteEnv2
# game-v2, internally determined pi_0, returns I[l], S[l] and S_0, continuous action space, mainly single class 
from gym.envs.ww.toy_game_2 import ToyDiscreteEnv3
# game-v3, return saliency location and I[l], mainly multiple classes, discrete action space
# TODO: support continuous action space 
from gym.envs.ww.toy_game_3 import ToyDiscreteEnv4

from gym.envs.ww.toy_game_4 import ToyDiscreteEnv5
