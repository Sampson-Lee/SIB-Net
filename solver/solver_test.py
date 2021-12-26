# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 21:40:17 2018

@author: lixinpeng
"""

import os, re, time, datetime, copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
import torchvision.utils as vutils
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2
from tensorboardX import SummaryWriter
import pdb
from IPython import embed
from torchnet import meter
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from shutil import copyfile
from sklearn.metrics import precision_recall_curve, average_precision_score

from auxi.loss_BCE import BCE_disc
from auxi.loss_MSE import MSE_disc, MSE_cont

disc_class = ['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning']

def denorm(img):
    img = img.transpose((1,2,0))*0.5 + 0.5
    img = np.uint8(255*img)
    return img

def cv2_heatmap(fmap, size):
    # average all channel and normalize
    fmap = np.mean(fmap, axis=0)
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
    fmap = cv2.resize(fmap, size, interpolation = cv2.INTER_CUBIC)
    fmap = (255 * fmap).astype(np.uint8)
    heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap[np.where(fmap <= 100)] = 0
    return heatmap

def cv2_bboxs(image, bbox):
    cv2.rectangle(image, tuple(bbox.astype(int)[0:2]), tuple(bbox.astype(int)[2:4]), (255,0,0), 2)
    return image

def test(model, data_loader, dataset, config):
    start = 1
    if config.resume_dir:
        model.load_state_dict(torch.load(config.resume_dir))
        print('resume successfully!')
    writer = SummaryWriter(log_dir=config.save_dir+'/'+time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time())))
    writer.add_text('Config', config.get_log())
    
    best_mAP_test = 0.0
    print(config.save_dir+'/'+config.get_name())
    for mode in ['test']:
        model.train(False)  # Set model to evaluate mode

        plotmap_list = []; label_list = []; score_list = []
        iter_ = 0
        for [data, label] in tqdm(data_loader[mode]):
            torch.set_grad_enabled(False)
            iter_ += 1
            # if iter_==3: break

            label['disc'] = label['disc'].cuda(); label['cont'] = label['cont'].cuda()
            data['image_scene'] = data['image_scene'].cuda(); data['image_body'] = data['image_body'].cuda(); data['image_head'] = data['image_head'].cuda()
            # out_list, fea_a, fea_b = model(data, mode)
            out_list, fea_a = model(data, mode); fea_b = fea_a
            out = F.sigmoid(out_list[0])
            out_disc = out
            
            # plot feature map for head
            fmaps = fea_a['ly4']['head']
            for img_index in range(fmaps.shape[0]):
                image_head = denorm(data['image_head'][img_index].cpu().numpy())
                fhmap_head_a = cv2_heatmap(fea_a['ly4']['head'][img_index].cpu().data.numpy(), size=image_head.shape[:2])
                fhmap_head_a = cv2.addWeighted(image_head, 0.7, fhmap_head_a, 0.3, 0)
                fhmap_head_b = cv2_heatmap(fea_b['ly4']['head'][img_index].cpu().data.numpy(), size=image_head.shape[:2])
                fhmap_head_b = cv2.addWeighted(image_head, 0.7, fhmap_head_b, 0.3, 0)
                plot_head = np.concatenate([image_head, fhmap_head_a, fhmap_head_b], axis=1)
                
                image_body = denorm(data['image_body'][img_index].cpu().numpy())
                fhmap_body_a = cv2_heatmap(fea_a['ly4']['body'][img_index].cpu().data.numpy(), size=image_body.shape[:2])
                fhmap_body_a = cv2.addWeighted(image_body, 0.7, fhmap_body_a, 0.3, 0)
                fhmap_body_b = cv2_heatmap(fea_b['ly4']['body'][img_index].cpu().data.numpy(), size=image_body.shape[:2])
                fhmap_body_b = cv2.addWeighted(image_body, 0.7, fhmap_body_b, 0.3, 0)
                plot_body = np.concatenate([image_body, fhmap_body_a, fhmap_body_b], axis=1)
                
                image_scene = denorm(data['image_scene'][img_index].cpu().numpy())
                fhmap_scene_a = cv2_heatmap(fea_a['ly4']['scene'][img_index].cpu().data.numpy(), size=image_scene.shape[:2])
                fhmap_scene_a = cv2.addWeighted(image_scene, 0.7, fhmap_scene_a, 0.3, 0)
                fhmap_scene_b = cv2_heatmap(fea_b['ly4']['scene'][img_index].cpu().data.numpy(), size=image_scene.shape[:2])
                fhmap_scene_b = cv2.addWeighted(image_scene, 0.7, fhmap_scene_b, 0.3, 0)
                plot_scene = np.concatenate([image_scene, fhmap_scene_a, fhmap_scene_b], axis=1)
                
                plot_img = np.concatenate([plot_head, plot_body, plot_scene], axis=0)
                save_path = os.path.join(config.save_dir, 'plot_heatmap')
                if not os.path.isdir(save_path): os.makedirs(save_path)
                cv2.imwrite(save_path+'/plot_img{}_ly4.jpg'.format(img_index+config.batch_size*(iter_-1)), 
                        cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))

            # score_list.append(out_disc.data.cpu().numpy())
            # label_list.append(label['disc'].gt(0).float().data.cpu().numpy())

        # plotmap_list = torch.cat(plotmap_list, dim=0).cpu().data.numpy()
        # # for img_index in range(plotmap_list.shape[0]):
        # for img_index in range(5):
        #     # for cha_index in range(plotmap_list.shape[1]):
        #     for cha_index in range(100):
        #         # embed()
        #         # pdb.set_trace()
        #         # plotmap_list[img_index,cha_index,:,:3] = 0
        #         plotmap = cv2_heatmap(plotmap_list[img_index,cha_index,:,:])
        #         save_path = os.path.join(config.save_dir, 'img_index'+str(img_index))
        #         if not os.path.isdir(save_path): os.makedirs(save_path)
        #         cv2.imwrite(save_path+'/cha_index{}.jpg'.format(cha_index), cv2.cvtColor(plotmap, cv2.COLOR_RGB2BGR))

        # score_list = np.concatenate(score_list, axis=0)
        # label_list = np.concatenate(label_list, axis=0)
        # img_list = np.array(dataset[mode].image_list)
        # # embed()
        # personbbox_list = np.array(dataset[mode].personbbox_list)
        # headbbox_list = np.array(dataset[mode].headbbox_list)
        # # embed()
        # # dataset[mode].image_list.index('/home/xinpeng/dataset/EMOTIC/emotic/emodb_small/images/0diuuuywatx5searpe.jpg')

        # # save error images
        # for class_index in range(score_list.shape[1]):
        #     sorted_ind = np.argsort(score_list[:,class_index])[::-1]
        #     sorted_label = label_list[:,class_index][sorted_ind]
        #     sorted_img = img_list[sorted_ind]
        #     sorted_headbbox = headbbox_list[sorted_ind]
        #     sorted_personbbox = personbbox_list[sorted_ind]

        #     print('process {} now'.format(disc_class[class_index]))
        #     # for img_idx, label in enumerate(sorted_label[:int(sorted_label.sum())]):
        #     topnum = 1
        #     for img_idx, label in enumerate(sorted_label[:topnum]):
        #         if label == 0:
        #             src_img = sorted_img[img_idx]
        #             print(src_img)
        #             dst_img = os.path.join(config.save_dir, 'top1', disc_class[class_index], src_img.split('/')[-1])
        #             if not os.path.isdir(os.path.dirname(dst_img)): os.makedirs(os.path.dirname(dst_img))
                    
        #             # copyfile(src_img, dst_img)
        #             personbbox = sorted_personbbox[img_idx]
        #             headbbox = sorted_headbbox[img_idx]
        #             image = cv2.imread(src_img)
        #             image = cv2_bboxs(image, personbbox)
        #             image = cv2_bboxs(image, headbbox)
        #             cv2.imwrite(dst_img, image)
            
        #     topnum = 5
        #     for img_idx, label in enumerate(sorted_label[:topnum]):
        #         if label == 0:
        #             src_img = sorted_img[img_idx]
        #             print(src_img)
        #             dst_img = os.path.join(config.save_dir, 'top5', disc_class[class_index], src_img.split('/')[-1])
        #             if not os.path.isdir(os.path.dirname(dst_img)): os.makedirs(os.path.dirname(dst_img))
        
        #             # copyfile(src_img, dst_img)
        #             personbbox = sorted_personbbox[img_idx]
        #             headbbox = sorted_headbbox[img_idx]
        #             image = cv2.imread(src_img)
        #             image = cv2_bboxs(image, personbbox)
        #             image = cv2_bboxs(image, headbbox)
        #             cv2.imwrite(dst_img, image)
        
        #     topnum = 10
        #     for img_idx, label in enumerate(sorted_label[:topnum]):
        #         if label == 0:
        #             src_img = sorted_img[img_idx]
        #             print(src_img)
        #             dst_img = os.path.join(config.save_dir, 'top10', disc_class[class_index], src_img.split('/')[-1])
        #             if not os.path.isdir(os.path.dirname(dst_img)): os.makedirs(os.path.dirname(dst_img))
                    
        #             # copyfile(src_img, dst_img)
        #             personbbox = sorted_personbbox[img_idx]
        #             headbbox = sorted_headbbox[img_idx]
        #             image = cv2.imread(src_img)
        #             image = cv2_bboxs(image, personbbox)
        #             image = cv2_bboxs(image, headbbox)
        #             cv2.imwrite(dst_img, image)
            
        #     topnum = 20
        #     for img_idx, label in enumerate(sorted_label[:topnum]):
        #         if label == 0:
        #             src_img = sorted_img[img_idx]
        #             print(src_img)
        #             dst_img = os.path.join(config.save_dir, 'top20', disc_class[class_index], src_img.split('/')[-1])
        #             if not os.path.isdir(os.path.dirname(dst_img)): os.makedirs(os.path.dirname(dst_img))
                    
        #             # copyfile(src_img, dst_img)
        #             personbbox = sorted_personbbox[img_idx]
        #             headbbox = sorted_headbbox[img_idx]
        #             image = cv2.imread(src_img)
        #             image = cv2_bboxs(image, personbbox)
        #             image = cv2_bboxs(image, headbbox)
        #             cv2.imwrite(dst_img, image)
            
        #     topnum = 30
        #     for img_idx, label in enumerate(sorted_label[:topnum]):
        #         if label == 0:
        #             src_img = sorted_img[img_idx]
        #             print(src_img)
        #             dst_img = os.path.join(config.save_dir, 'top30', disc_class[class_index], src_img.split('/')[-1])
        #             if not os.path.isdir(os.path.dirname(dst_img)): os.makedirs(os.path.dirname(dst_img))
                    
        #             # copyfile(src_img, dst_img)
        #             personbbox = sorted_personbbox[img_idx]
        #             headbbox = sorted_headbbox[img_idx]
        #             image = cv2.imread(src_img)
        #             image = cv2_bboxs(image, personbbox)
        #             image = cv2_bboxs(image, headbbox)
        #             cv2.imwrite(dst_img, image)
            
        #     topnum = 40
        #     for img_idx, label in enumerate(sorted_label[:topnum]):
        #         if label == 0:
        #             src_img = sorted_img[img_idx]
        #             print(src_img)
        #             dst_img = os.path.join(config.save_dir, 'top40', disc_class[class_index], src_img.split('/')[-1])
        #             if not os.path.isdir(os.path.dirname(dst_img)): os.makedirs(os.path.dirname(dst_img))
                    
        #             # copyfile(src_img, dst_img)
        #             personbbox = sorted_personbbox[img_idx]
        #             headbbox = sorted_headbbox[img_idx]
        #             image = cv2.imread(src_img)
        #             image = cv2_bboxs(image, personbbox)
        #             image = cv2_bboxs(image, headbbox)
        #             cv2.imwrite(dst_img, image)
        # # calculate mAP
        # # embed()
        # ap_list=[]; f1_list=[]; th_list=[]
        # precision = {}; recall = {}
        # for class_index in range(score_list.shape[1]):
        #     precision[class_index], recall[class_index], thresholds = precision_recall_curve(label_list[:,class_index], score_list[:,class_index])
        #     # plot_pr_img(precision[class_index], recall[class_index], os.path.join(config.save_dir, disc_class[class_index]))
            
        #     average_precision = average_precision_score(label_list[:,class_index], score_list[:,class_index])
            
        #     f1 = 2*precision[class_index][2:]*recall[class_index][2:]/(precision[class_index][2:]+recall[class_index][2:])

        #     m_idx = np.argmax(f1)
        #     m_thresh = thresholds[2+m_idx]
            
        #     print("%s %f %f %f" % (disc_class[class_index], average_precision, f1[m_idx], m_thresh))
        #     ap_list.append(average_precision)
        #     f1_list.append(f1[m_idx])
        #     th_list.append(m_thresh)

        #     fig = plt.figure(figsize=(20,20))
        #     plt.title('Precision/Recall Curve')
        #     plt.xlabel('Recall')
        #     plt.ylabel('Precision')
        #     plt.plot(recall[class_index], precision[class_index])
        #     fig.canvas.draw()
        #     fig_arr = np.array(fig.canvas.renderer._renderer)
        #     plt.close()
            
        #     save_path = os.path.join(config.save_dir, disc_class[class_index]+'_pr.png')
        #     cv2.imwrite(save_path, cv2.cvtColor(fig_arr, cv2.COLOR_BGRA2RGB))

        # ap_list = np.array(ap_list)
        # print(ap_list.mean())

        # if mode == 'test' and ap_list.mean() > best_mAP_test:
        #     best_mAP_test = ap_list.mean()
        #     best_mAP_model = copy.deepcopy(model.state_dict())

    print('best accuracy in test is : ', best_mAP_test)

    writer.export_scalars_to_json(os.path.join(config.save_dir, "log.json"))
    writer.close()

    print('train and validate model successfully')
    return [best_mAP_test], []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

def solver_custom(config):
    print(config.save_dir+'/'+config.get_name())

    setup_seed(config.seed)
    exec('from data.{} import load_data'.format(config.datasetname), globals())
    data_loader, dataset = load_data(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    exec('from net.{} import net'.format(config.modelname), globals())
    model = net(config).cuda()
    
    if len(config.device.split(','))>1:
        print('model uses multigpu!')
        model = nn.DataParallel(model)

    best_log, best_model = test(model, data_loader, dataset, config)

    return best_log
