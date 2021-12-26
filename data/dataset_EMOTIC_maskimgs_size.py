# -*- coding: utf-8 -*-
"""
Created on Sat Dec 2019

@author: lixinpeng
"""

import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import cv2
from IPython import embed
import scipy.io as sio
import math
import os, collections

import torchvision.utils as vutils
# from .custom_sampler import BalancedBatchSampler

from tqdm import tqdm
import torchfile
import copy
import torchvision.transforms as transforms
from torchvision.transforms import Resize, RandomCrop, CenterCrop, ToTensor, Normalize

disc_class = ['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning']
cont_class = ['Valence','Arousal','Dominance']

"""
dataset
"""
class EMOTICDataset(data.Dataset):
    def __init__(self, config, mode='train', batch_size=64):
        self.meta_dir = config['meta_dir']
        self.img_dir = config['img_dir']
        self.mode = mode

        self.transform = {
            'train_head': transforms.Compose([
                Resize((230,230)),
                RandomCrop((224,224)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            
            'train_person': transforms.Compose([
                Resize((230,230)),
                RandomCrop((224,224)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            
            'train_scene': transforms.Compose([
                Resize((230,230)),
                RandomCrop((224,224)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            
            'val_head': transforms.Compose([
                Resize((230,230)),
                RandomCrop((224,224)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            
            'val_person': transforms.Compose([
                Resize((230,230)),
                RandomCrop((224,224)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            
            'val_scene': transforms.Compose([
                Resize((230,230)),
                RandomCrop((224,224)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        }

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        # embed()

        print('Analysing data distribution..!')
        self.statistics()
        print('Finished data distribution..!')

        # embed()
        if mode=='train':
            self.num_data = len(self.image_list)
            # self.num_data = batch_size*5e4/20
            # self.num_data = max(self.numPclass)*len(disc_class)
            print('FERDataset has {} {} images, and augment to {} images'.format(len(self.image_list), self.mode, self.num_data))
        else:
            self.num_data = len(self.image_list)
            print('FERDataset has {} {} images'.format(self.num_data, self.mode))

    def preprocess(self):
        self.image_list = []
        self.disc_label_list = []
        self.cont_label_list = []
        self.personbbox_list = []
        self.headbbox_list = []

        annotations = torchfile.load(self.meta_dir+'DiscreteContinuousAnnotations26_'+self.mode+'.t7')
        for item in tqdm(annotations):
            # embed()
            filename = str(item[b'filename'], encoding='utf-8')
            folder = str(item[b'folder'], encoding='utf-8')
            # filename = str(item[b'filename']).encode('utf-8')
            # folder = str(item[b'folder']).encode('utf-8')

            head_bbox = item[b'head_bbox']
            body_bbox = item[b'body_bbox']

            workers = item[b'workers']
            disc_labels = np.zeros(len(disc_class))
            cont_labels = np.zeros(len(cont_class))
            for worker in workers:
                for cate in worker[b'labels']:
                    disc_labels[cate-1] = disc_labels[cate-1] + 1
                cont_labels = cont_labels + np.array(list(worker[b'continuous'].values()))/10
            disc_labels = np.clip(disc_labels/disc_labels.max(), 0, 1)
            cont_labels = cont_labels/len(workers)
            # print(self.img_dir, folder, filename)
            self.image_list.append(self.img_dir+folder+'/'+filename)
            self.disc_label_list.append(disc_labels)
            self.cont_label_list.append(cont_labels)
            self.personbbox_list.append(body_bbox)
            self.headbbox_list.append(head_bbox)

    def statistics(self):
        fig = plt.figure(figsize=(20,20))
        labelMatrix = np.array(self.disc_label_list)
        numPclass = labelMatrix.sum(axis=0).astype(int)
        self.numPclass = numPclass

        # plot distribution
        sorted_numPclass = np.sort(numPclass)[::-1]
        ind = np.arange(len(disc_class))    # the x locations for the groups
        plt.bar(ind, sorted_numPclass)
        plt.xticks(ind, disc_class, fontsize=5)
        plt.xlabel('class');plt.ylabel('number')

        sorted_indices = np.argsort(-numPclass)
        for ind_ind, ind_ in enumerate(sorted_indices):
            print(disc_class[ind_], numPclass[ind_])
            plt.text(ind_ind, numPclass[ind_]+0.05, '{}'.format(numPclass[ind_]), ha='center', va='bottom', fontsize=7)

        fig.canvas.draw()
        fig_arr = np.array(fig.canvas.renderer._renderer)
        plt.close()
        cv2.imwrite(os.path.dirname(self.meta_dir)+'/EMOTIC_datavisaul_'+self.mode+'.jpg', \
                    cv2.cvtColor(fig_arr, cv2.COLOR_BGRA2RGB))

        # calculate weights
        # self.weights_list = list(map(lambda x: 1/math.log(x+1.2), self.numPclass/labelMatrix.shape[0]))
        # self.weights_list = list(map(lambda x: log(x), numPclass))
        # self.weights_list = list(map(lambda x: x/total, numPclass))
        self.weights_list = list(map(lambda x: math.log(x)/10, numPclass))

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        if image.mode=='L': image = image.convert('RGB')
        if image.mode=='RGBA': image = image.convert('RGB')

        # different image
        personbbox = self.personbbox_list[index]
        image_body = image.crop(personbbox)

        headbbox = self.headbbox_list[index]
        image_head = image.crop(headbbox)

        img_arr = np.array(image)
        img_arr[headbbox[1]:headbbox[3],headbbox[0]:headbbox[2],:] = 0

        image_mask_head = Image.fromarray(img_arr)
        image_body_mask_head = image_mask_head.crop(personbbox)

        img_arr[personbbox[1]:personbbox[3],personbbox[0]:personbbox[2],:] = 0

        image_mask_body = Image.fromarray(img_arr)
        if self.mode=='train':
            data = {'image_scene': self.transform['train_scene'](image_mask_body),
                    'image_body': self.transform['train_person'](image_body_mask_head),
                    'image_head': self.transform['train_head'](image_head)}
        else:
            data = {'image_scene': self.transform['val_scene'](image_mask_body),
                    'image_body': self.transform['val_person'](image_body_mask_head),
                    'image_head': self.transform['val_head'](image_head)}

        label = {'disc':self.disc_label_list[index], 'cont':self.cont_label_list[index]}
        return data, label

    def __len__(self):
        return self.num_data

#########
# data loader
#########

def denorm(img):
    img = img.transpose((1,2,0))*0.5 + 0.5
    img = np.uint8(255*img)
    return img

def cv2_landmarks(image, landmarks):
    for idx, point in enumerate(landmarks):
        cv2.circle(image, center=(point[0], point[1]), radius=2, color=(255, 0, 0), thickness=-1)
        # cv2.putText(image, str(idx+1), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
    return image

def cv2_bboxs(image, bbox):
    cv2.rectangle(image, tuple(bbox.astype(int)[0:2]), tuple(bbox.astype(int)[2:4]), (255,0,0), 2)
    return image

def load_data(config):

    Dataset = {
            'train': EMOTICDataset(config.EMOTIC, mode='train', batch_size=config.batch_size),
            'val': EMOTICDataset(config.EMOTIC,  mode='val', batch_size=32),
            'test': EMOTICDataset(config.EMOTIC, mode='test', batch_size=32),
    }

    data_loader = {
            'train': data.DataLoader(Dataset['train'], batch_size= config.batch_size, \
                    shuffle=True, num_workers=3, worker_init_fn = np.random.seed(0)), 
            'val': data.DataLoader(Dataset['val'], batch_size= config.batch_size, shuffle= False, num_workers=3, worker_init_fn = np.random.seed(0)),
            'test': data.DataLoader(Dataset['test'], batch_size= config.batch_size, shuffle= False, num_workers=3, worker_init_fn = np.random.seed(0)), 
    }

    input_data, _ = next(iter(data_loader['train']))
    image_scene, image_body, image_head = input_data['image_scene'], input_data['image_body'], input_data['image_head']
    vutils.save_image(image_scene, config.save_dir+'/samples_train_image_scene.png', nrow=8, padding=2, normalize=True)
    vutils.save_image(image_body, config.save_dir+'/samples_train_image_body.png', nrow=8, padding=2, normalize=True)
    vutils.save_image(image_head, config.save_dir+'/samples_train_image_head.png', nrow=8, padding=2, normalize=True)

    # from IPython import embed; embed(); exit()
    return data_loader, Dataset
