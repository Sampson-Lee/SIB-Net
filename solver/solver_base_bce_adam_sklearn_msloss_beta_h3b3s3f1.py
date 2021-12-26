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
# from IPython import embed
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchnet import meter
from auxi.loss_BCE import BCE_disc, BCE_disc_sm, BCE_disc_sm_beta
from auxi.loss_MSE import MSE_disc, MSE_cont

disc_class = ['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning']

def train(model, data_loader, dataset, config):
    start = 1
    if config.resume_dir:
        pattern = re.compile(r'epoch(?P<start>\d+).pth')
        m = re.search(pattern, config.resume_dir)
        if m: start = int(m.groupdict['start'])
        try:
            model.load_state_dict(torch.load(config.resume_dir))
            print('resume successfully!')
        except:
            pass
    writer = SummaryWriter(log_dir=config.save_dir+'/'+time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time())))
    writer.add_text('Config', config.get_log())
    lossMeter = meter.AverageValueMeter()

    paras_upper = []
    for key, value in model.named_parameters():
        # print(key)
        if 'fc3' in key or 'mlp' in key or 'att' in key or 'ffm' in key or 'lowup' in key or 'imm' in key or 'sfm' in key or 'cls' in key:
            print(key)
            paras_upper.append(value)
    paras_base = list(set(model.parameters())-set(paras_upper))
    paras = [{'params':paras_upper, 'lr':config.up_lr, 'weight_decay':config.weight_decay},
            {'params':paras_base, 'lr':config.low_lr, 'weight_decay':config.weight_decay}]

    print('using '+config.optim+' and '+config.lr_policy)
    optimizer = optim.Adam(paras, betas=(config.beta1_Adam, config.beta2_Adam))
    milestones = [int(config.epochs*(1-0.5**multi)) for multi in range(1,4)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion_disc_sm_three = BCE_disc_sm_beta(lb_sm=0.3)
    criterion_disc_sm_one = BCE_disc_sm_beta(lb_sm=0.1)
    criterion_disc = BCE_disc()

    print('*'*5)
    print('training begin ')
    print('*'*5)
    since = time.time()

    best_mAP_test = 0.0
    for epoch in range(start, config.epochs+1):
        print(config.save_dir+'/'+config.get_name())
        for mode in ['train', 'val', 'test']:
            if mode == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            lossMeter.reset()
            score_list = []; label_list = []
            for i, [data, label] in enumerate(data_loader[mode]):
                if mode == 'train': torch.set_grad_enabled(True)
                else: torch.set_grad_enabled(False)
                
                label['disc'] = label['disc'].cuda(); label['cont'] = label['cont'].cuda()
                data['image_scene'] = data['image_scene'].cuda(); data['image_body'] = data['image_body'].cuda(); data['image_head'] = data['image_head'].cuda()

                optimizer.zero_grad()
                out_list = model(data, mode)
                
                loss = (criterion_disc(F.sigmoid(out_list[0]), label['disc'].float())).sum()
                loss += (criterion_disc_sm_one(F.sigmoid(out_list[1]), label['disc'].float())).sum()
                for out in out_list[2:]:
                    out = F.sigmoid(out)
                    out_disc = out
                    loss_disc = criterion_disc_sm_three(out_disc, label['disc'].float())
                    loss += loss_disc.sum()

                lossMeter.add(loss.item())
                out_disc = out_list[0]
                score_list.append(out_disc.data.cpu().numpy())
                label_list.append(label['disc'].gt(0).float().data.cpu().numpy())

                if mode == 'train':
                    loss.backward()
                    optimizer.step()

                    if (i+1)%20==0:
                        time_elapsed = time.time() - since
                        time_elapsed = str(datetime.timedelta(seconds=time_elapsed))
                        print('Elapsed [{}] Epoch {}/{} Iter {}/{} EMloss {:.8f} '.format(time_elapsed, epoch, 
                            config.epochs, i+1, len(data_loader[mode]), loss.item()))
                        
                        n_iter = i+1+(epoch-1)*len(data_loader[mode])
                        writer.add_scalars('loss in every iter for {} '.format(config.modelname), {'EMloss':loss.item()}, n_iter)
                        writer.add_scalars('class loss in every iter for {} '.format(config.modelname), {disc_class[idx]+' loss':loss_disc[idx].item() for idx in range(loss_disc.size(0))}, n_iter)
                        writer.add_scalars('lr in every iter for {} '.format(config.modelname), {'para_group{}'.format(idx):param_group['lr'] for idx, param_group in enumerate(optimizer.param_groups)}, n_iter)

            scheduler.step()

            label_list = np.concatenate(label_list, axis=0); score_list = np.concatenate(score_list, axis=0)
            ap_list=[]; f1_list=[]; th_list=[]
            precision = {}; recall = {}
            for class_index in range(score_list.shape[1]):
                precision[class_index], recall[class_index], thresholds = precision_recall_curve(label_list[:,class_index], score_list[:,class_index])
                f1 = 2*precision[class_index][2:]*recall[class_index][2:]/(precision[class_index][2:]+recall[class_index][2:])
                average_precision = average_precision_score(label_list[:,class_index], score_list[:,class_index])
                m_idx = np.argmax(f1)
                m_thresh = thresholds[2+m_idx]
                print("%s %f %f %f" % (disc_class[class_index], average_precision, f1[m_idx], m_thresh))
                ap_list.append(average_precision)
                f1_list.append(f1[m_idx])
                th_list.append(m_thresh)
            ap_list = np.array(ap_list)

            print('*'*5)
            print(mode + ' Epoch: {} Loss:'.format(epoch))
            print(lossMeter.value())
            print(mode + ' Epoch: {} mAP:'.format(epoch))
            print(ap_list.mean())

            writer.add_scalars('loss and mAP of modes in every Epoch for {} '.format(config.modelname), {mode+'loss': lossMeter.value()[0], mode+'mAP': ap_list.mean()}, epoch)
            writer.add_scalars('every class AP of modes in every Epoch for {} '.format(config.modelname), {mode+'{} AP'.format(disc_class[idx]): ap for idx, ap in enumerate(ap_list)}, epoch)
            writer.add_scalars('every class F1 of modes in every Epoch for {} '.format(config.modelname), {mode+'{} F1'.format(disc_class[idx]): f1 for idx, f1 in enumerate(f1_list)}, epoch)
            writer.add_scalars('every class TH of modes in every Epoch for {} '.format(config.modelname), {mode+'{} TH'.format(disc_class[idx]): th for idx, th in enumerate(th_list)}, epoch)

            if mode == 'test' and ap_list.mean() > best_mAP_test:
                best_mAP_test = ap_list.mean()
                best_mAP_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best accuracy in test is : ', best_mAP_test)

    writer.export_scalars_to_json(os.path.join(config.save_dir, "log.json"))
    writer.close()

    torch.save(best_mAP_model, os.path.join(config.save_dir, 'seed_{}_best_mAP_model_{:.4f}'.format(config.seed, best_mAP_test)+'.pth'))

    print('train and validate model successfully')
    return [best_mAP_test], [best_mAP_model]

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
    # model = net(config)

    if len(config.device.split(','))>1:
        print('model uses multigpu!')
        model = nn.DataParallel(model)

    best_log, best_model = train(model, data_loader, dataset, config)

    return best_log
