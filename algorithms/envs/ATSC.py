# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 19:00:22 2022

@author: 86153
"""


import gym
import numpy as np
from .NCS.large_grid_env import LargeGridEnv
from .NCS.real_net_env import RealNetEnv
from gym.spaces import Box, Discrete
import configparser
import os
import pdb
# from .NCS.envs.large_grid_data.build_file import main
# main()
# from ..utils import listStack




def Grid_Env():
    # return GridWrapper('NCS/config/config_ma2c_nc_grid.ini', bias=0, std=100)
    config = configparser.ConfigParser()
    # config.read('D:/A_RL/MB-MARL/algorithms/envs/NCS/config/config_ma2c_nc_grid.ini')
    config.read('algorithms/envs/NCS/config/config_ma2c_nc_grid.ini')
    env = LargeGridEnv(config['ENV_CONFIG'])  
    return env

def Monaco_Env():
    # return GridWrapper('NCS/config/config_ma2c_nc_grid.ini', bias=0, std=100)
    config = configparser.ConfigParser()
    config.read('algorithms/envs/NCS/config/config_ma2c_nc_net.ini')
    env = RealNetEnv(config['ENV_CONFIG'])  
    return env