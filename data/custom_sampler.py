# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
import pdb

disc_class = ['Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning']

class BalancedBatchSampler(Sampler):
    """
    BatchSampler - from a ImageFloderLoader dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples): 

        self.labels = np.array(dataset.disc_label_list)
        # print(self.labels)
        self.labels_set = list(set(np.arange(n_classes)))
        self.label_to_indices = {label: np.where(self.labels[:,label] == 1)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        indices = []

        while self.count + self.batch_size < len(self.dataset): # sample a batch
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)

            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0  

            self.count += self.n_classes * self.n_samples

        return iter(indices)

    def __len__(self):
        return len(self.dataset) // self.batch_size

def IRLbl(labels):
	# imbalance ratio per label
	# Args:
	#	 labels is a 2d numpy array, each row is one instance, each column is one class; the array contains (0, 1) only
	N, C = labels.shape
	pos_nums_per_label = np.sum(labels, axis=0)
	max_pos_nums = np.max(pos_nums_per_label)
	return max_pos_nums/pos_nums_per_label

def MeanIR(labels):
	IRLbl_VALUE = IRLbl(labels)
	return np.mean(IRLbl_VALUE)

class ImbalancedDatasetSampler_ML(Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
		callback_get_label func: a callback-like function which takes two arguments - dataset and index
	"""

	def __init__(self, dataset, indices=None, num_samples=None, Preset_MeanIR_value= 2.0, 
		               max_clone_percentage=5000, sample_size=32):

		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) \
			if num_samples is None else num_samples
		pdb.set_trace()
		all_labels = np.array(dataset.disc_label_list)
		MeanIR_value = MeanIR(all_labels) if Preset_MeanIR_value ==0 else Preset_MeanIR_value
		IRLbl_value = IRLbl(all_labels)
		N, C = all_labels.shape
		indices_per_class = {}
		minority_classes = []
		maxSamplesToClone = N / 100 * max_clone_percentage
		for i in range(C):
			ids = all_labels[:,i] == 1
			indices_per_class[i] = [ii for ii, x in enumerate(ids) if x ]
			if IRLbl_value[i] > MeanIR_value:
				minority_classes.append(i)
		new_all_labels = all_labels
		oversampled_ids = []
		for i in minority_classes:
			while True:
				pick_id = list(np.random.choice(indices_per_class[i], sample_size))
				indices_per_class[i].extend(pick_id)
				# recalculate the IRLbl_value
				new_all_labels = np.concatenate([new_all_labels, all_labels[pick_id]], axis=0)
				oversampled_ids.extend(pick_id)
				if IRLbl(new_all_labels)[i] <= MeanIR_value or len(oversampled_ids)>=maxSamplesToClone :
					break
				print("oversample length:{}".format(len(oversampled_ids)), end='\r')
			if len(oversampled_ids) >=maxSamplesToClone:
				break

		weights = np.array([1.0/len(self.indices)] * len(self.indices))
		unique, counts =  np.unique(oversampled_ids, return_counts=True)
		for i, n in zip(unique, counts):
			weights[i] = weights[i]*n
		self.weights = torch.DoubleTensor(weights)

        # plot distribution
		numPclass_exp = new_all_labels.sum(axis=0).astype(int)
		sorted_numPclass_exp = np.sort(numPclass_exp)[::-1]
		fig = plt.figure(figsize=(20,20))
		ind = np.arange(len(disc_class))    # the x locations for the groups
		plt.bar(ind, sorted_numPclass_exp)
		plt.xticks(ind, disc_class, fontsize=5)
		plt.xlabel('class');plt.ylabel('number')

		sorted_indices = np.argsort(-numPclass_exp)
		for ind_ind, ind_ in enumerate(sorted_indices):
			print(disc_class[ind_], numPclass_exp[ind_])
			plt.text(ind_ind, numPclass_exp[ind_]+0.05, '{}'.format(numPclass_exp[ind_]), ha='center', va='bottom', fontsize=7)

		fig.canvas.draw()
		fig_arr = np.array(fig.canvas.renderer._renderer)
		plt.close()
		cv2.imwrite('/data3/xinpeng/EMOTIC/annotations/EMOTIC_datavisaul_train_augmentation.jpg', \
                    cv2.cvtColor(fig_arr, cv2.COLOR_BGRA2RGB))

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples


class ImbalancedDatasetSampler_VA(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
		callback_get_label func: a callback-like function which takes two arguments - dataset and index
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) \
			if num_samples is None else num_samples
		 
		all_labels = dataset._get_all_label()
		N, C = all_labels.shape
		assert C == 2
		hist, x_edges, y_edges = np.histogram2d(all_labels[:, 0], all_labels[:, 1], bins=[20, 20])
		x_bin_id = np.digitize( all_labels[:, 0], bins = x_edges) - 1
		y_bin_id = np.digitize( all_labels[:, 1], bins = y_edges) - 1
		# for value beyond the edges, the function returns len(digitize_num), but it needs to be replaced by len(edges)-1
		x_bin_id[x_bin_id==20] = 20-1
		y_bin_id[y_bin_id==20] = 20-1
		weights = []
		for x, y in zip(x_bin_id, y_bin_id):
			assert hist[x, y]!=0
			weights += [1 / hist[x, y]] 
		
		self.weights = torch.DoubleTensor(weights)

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples


class ImbalancedDatasetSampler_SLML(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in tqdm(self.indices, total = len(self.indices)):
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            return dataset._data['label'][idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
