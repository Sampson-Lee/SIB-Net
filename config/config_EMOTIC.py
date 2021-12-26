import numpy as np
import datetime
import os
# from IPython import embed
import pdb

class config(object):
    def __init__(self):
        self.pretrainedmodel_dir = '/home/xinpeng/dataset/PreModel/pretrained_imagenet_resnet34.pth'

        self.datasetname = 'dataset_EMOTIC_maskimgs_size'
        if os.getcwd().find('xian')!=-1:
            self.EMOTIC = {
                'meta_dir': '/home/xian/Documents/xinpeng/EMOTIC/annotations/',
                'img_dir': '/home/xian/Documents/xinpeng/EMOTIC/emotic/',}
        else:
            self.EMOTIC = {
                'meta_dir': '/home/xinpeng/dataset/EMOTIC/annotations/',
                'img_dir': '/home/xinpeng/dataset/EMOTIC/emotic/',}

        self.datestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.modelname = 'net_HBS_solo_resnet18_score_fuse_FFM4_c2_nonlocal_ly24_mout_rnn_ly1_h1_ly4321'
        self.solvername = 'solver_base_bce_adam_sklearn_msloss_beta_h3b3s3f1'

        self.num_classes = 26
        print(os.getcwd().find('xian'))
        print(os.getcwd().find('lixinpeng'))
        if os.getcwd().find('xian')!=-1:
            self.save_dir = '/home/xian/Documents/xinpeng/Log_Scene'
            self.resume_dir_B = '/home/xian/Documents/xinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_B_solo_resnet18/seed_2019_best_mAP_model_0.2940.pth'
            self.resume_dir_H = '/home/xian/Documents/xinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_H_solo_resnet18/seed_2019_best_mAP_model_0.3268.pth'
            self.resume_dir_S = '/home/xian/Documents/xinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_S_solo_resnet18/seed_2019_best_mAP_model_0.2975.pth'

        elif os.getcwd().find('lixinpeng')!=-1:
            self.save_dir = '/home/lixinpeng/Log_Scene'
            self.resume_dir_B = '/home/lixinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_B_solo_resnet18/seed_2019_best_mAP_model_0.2940.pth'
            self.resume_dir_H = '/home/lixinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_H_solo_resnet18/seed_2019_best_mAP_model_0.3268.pth'
            self.resume_dir_S = '/home/lixinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_S_solo_resnet18/seed_2019_best_mAP_model_0.2975.pth'

        else:
            self.save_dir = '/home/xinpeng/dataset/Log_Scene'
            self.resume_dir_B = '/data3/xinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_B_solo_resnet18/seed_2019_best_mAP_model_0.2940.pth'
            self.resume_dir_H = '/data3/xinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_H_solo_resnet18/seed_2019_best_mAP_model_0.3268.pth'
            self.resume_dir_S = '/data3/xinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_S_solo_resnet18/seed_2019_best_mAP_model_0.2975.pth'
        self.resume_dir = '/home/xian/Documents/xinpeng/Log_Scene/dataset_EMOTIC_maskimgs_size/solver_base_bce_adam_sklearn/net_HBS_solo_resnet18_score_fuse/seed_2019_best_mAP_model_0.3348.pth'

        self.batch_size = 32
        self.crop_size = 224

        self.epochs = 10
        self.optim = 'Adam'
        self.lr_Adam = 0.0002
        self.beta1_Adam = 0.5
        self.beta2_Adam = 0.999
        self.momentum = 0.9
        self.weight_decay = 0.0005

        self.low_lr = 1e-4
        self.up_lr = 1e-3
        self.lr_policy = 'step_multistep'

        self.device = '0'
        self.seed = 2019

    def get_name(self):
        reg_str = ['lr', 'dir','size','device','beta','mom','wei','list','poli','pre','opt','Adam','epochs','name', 'datestr', 'EMOTIC']
        name_str = ''
        for name,value in vars(self).items():
            flag = 1
            for ind, reg in enumerate(reg_str):
                if reg in str(name):
                    flag = 0
                    break
            if flag: name_str += str(name)+'_'+str(value)+'-'
        return name_str

    def get_log(self):
        name_str = ''
        for name, value in vars(self).items():
            # print(name)
            name_str += str(name)+'_'+str(value)+'\n'
            # print(name_str)
        return name_str
