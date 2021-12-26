# -*- coding: utf-8 -*-
"""
Created on Sat Dec 2019

@author: lixinpeng
"""

import os, time, sys
import torch
import torch.nn as nn
import numpy as np
import shutil
# from IPython import embed
sys.path.append('.')
config_name = 'config_EMOTIC'

if __name__ == '__main__':

    exec('from config.{} import config'.format(config_name))
    config = config()

    config.save_dir = config.save_dir + '/' + config.datasetname + '/' + config.solvername + '/' + config.modelname
    exec('from solver.{} import solver_custom'.format(config.solvername))
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)
        shutil.copytree('./auxi/', config.save_dir+'/auxi')
    shutil.copyfile('./config/'+config_name+'.py', config.save_dir+'/'+config_name+'.py')
    shutil.copyfile('./data/'+config.datasetname+'.py', config.save_dir+'/'+config.datasetname+'.py')
    shutil.copyfile('./net/'+config.modelname+'.py', config.save_dir+'/'+config.modelname+'.py')
    shutil.copyfile('./solver/'+config.solvername+'.py', config.save_dir+'/'+config.solvername+'.py')

    mAP_test_list = []

    for i in range(3):
        config.fold = i
        config.seed += i
        log_rec = solver_custom(config)
        mAP_test_list.append(log_rec[0])

    log_file = os.path.join(config.save_dir, 'log_rec'+config.datestr+'.txt')
    f = open(log_file, 'w+')

    mAP_test_list = np.array(mAP_test_list)
    np.set_printoptions(precision=4, suppress=True)

    print(mAP_test_list)
    print(mAP_test_list.mean(), mAP_test_list.std())
    f.write('mAP test {} {} {} \n'.format(mAP_test_list, mAP_test_list.mean(), mAP_test_list.std()))
    f.close()
