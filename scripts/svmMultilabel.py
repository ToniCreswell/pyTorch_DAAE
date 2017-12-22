#Train and test SVM on several different labels

import sys
sys.path.append('../')

from function import prep_data, make_new_folder, plot_losses, save_input_args, shift_x, plot_norm_losses
from dataload import CELEBA 
from models import DAE, DIS_Z, LINEAR_SVM

import torch
from torch import optim

from torchvision.utils import make_grid, save_image
from torchvision import transforms, datasets

import argparse

from time import time

import os
from os.path import join

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

def svm_score(svm, y, x=None, enc=None, dae=None):
	'''
	EITHER
		take a data sample x AND a dae
	OR
		take an encoding
	and apply SVM and get score

	'''
	assert (x is not None) or (enc is not None)
	if enc is None:
		assert dae is not None
		enc = dae.encode(x)
	output = svm.forward(enc)
	score = svm.binary_class_score(output, y)
	return score

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='/data', type=str) #data/datasets/LabelSwap
	parser.add_argument('--batchSize', default=100, type=int)
	parser.add_argument('--maxEpochs', default=20, type=int)
	parser.add_argument('--nz', default=200, type=int)
	parser.add_argument('--fSize', default=64, type=int)  # multiple of filters to use
	parser.add_argument('--load_from', default=None, type=str)
	parser.add_argument('--c', type=float, default=0.01) #for training the linearSVM for eval
	parser.add_argument('--svmLR', type=float, default=2e-4)
	parser.add_argument('--Ntest', default=100, type=int)
	parser.add_argument('--gpuNo', required=True, type=int)
	parser.add_argument('--sigma', default=1.0, type=float)

	
	return parser.parse_args()

def train_svm(dae, svm, trainLoader, testLoader, exDir, lr, label):
	'''
	Data y is [0,1]
	For training SVM must be -1, 1
	To eval data put back to [0,1]

	To get loss use [-1,1] for train and test
	To get score use [0,1] for train and test
	'''
	print 'training svm...'
	dae.eval()
	optimSVM = optim.SGD(svm.parameters(), lr=lr) #optimizer  

	f = open(join(exDir, 'multiSVMppts.txt'), 'w')
	f.write('smvLR: %0.5f\nc: %0.5f\n' % (lr, svm.c))
	f.close() 


	svmLoss = {'train':[], 'test':[]}
	for epoch in range(opts.maxEpochs):
		epochLoss_svm=0
		svm.train()
		T = time()
		for i, data in enumerate(trainLoader):
			x, y = prep_data(data, useCUDA=svm.useCUDA)  #prep data as a var
			inputs = dae.encode(x)  #get encodings as input
			output = svm.forward(inputs)  #get output
			loss = svm.loss(output, y * 2 - 1)  #calc loss 
			optimSVM.zero_grad()  #zero grad
			loss.backward()  #backwards
			optimSVM.step()  #step
			epochLoss_svm+=loss.data[0]

			if i%100 == 0:
				print '[%d, %i] loss: %0.5f, time: %0.3f' % (epoch, i, epochLoss_svm/(i+1), time() - T)
		svm.save_params(exDir)
		svmLoss['train'].append(epochLoss_svm/(i+1))

		#test loss:
		svm.eval()
		xTest, yTest = prep_data(iter(testLoader).next(), useCUDA=svm.useCUDA)
		testInputs = dae.encode(xTest)
		testOutputs = svm.forward(testInputs)
		testLoss = svm.loss(testOutputs, yTest * 2 - 1)
		svmLoss['test'].append(testLoss.data[0])

		if epoch > 1:
			plot_losses(svmLoss, exDir=exDir, epochs=epoch+1, title=label+': SVM loss')

		#Do classification
		testScore = svm.binary_class_score(testOutputs, yTest) #has threshold as zero for testOutputs in [-1,1]
		trainScore = svm.binary_class_score(output, y) #has threshold as zero for output in [-1,1]
		f = open(join(exDir, 'svm_multi.txt'), 'w')
		f.write('%s :: trainScore: %f \ntestScore: %f ' \
		 % (label, trainScore.mean().data[0], testScore.mean().data[0]))
		f.close()
 	
	return svm


if __name__=='__main__':

	opts = get_args()

	#new folder to save SVM results
	svmDir = join(opts.load_from,'new_multiSVM')
	try:
		os.mkdir(svmDir)
	except OSError:
		print 'already exsits'

	#load model
	dae = DAE(nz=opts.nz, imSize=64, fSize=opts.fSize, sigma=opts.sigma, multimodalZ=False) #sigma=level of corruption
	svm = LINEAR_SVM(nz=opts.nz, c=opts.c) #model

	if dae.useCUDA:
		torch.cuda.set_device(opts.gpuNo)
		print 'using gpu:', torch.cuda.current_device()
		dae.cuda()
		svm.cuda()
	dae.eval()

	dae.load_params(opts.load_from)
	print 'models loaded'

	f = open(join(svmDir, 'multiSVMoutputs.txt'), 'a')
	f.write('-----new experiment: c= %f, batchSize= %f, epochs= %d -----' % (opts.c, opts.batchSize, opts.maxEpochs))
	f.close()

	attributes = ['Arched_Eyebrows', 'Attractive', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'No_Beard', 'Wavy_Hair', 'Wearing_Hat', 'Wearing_Lipstick']
	for i in range(len(attributes)):
		label = attributes[i]
		#Load data
		print 'Prepare data loaders...'
		transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
		trainDataset = CELEBA(root=opts.root, train=True, transform=transforms.ToTensor(), label=label, Ntest=1000)
		trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

		testDataset = CELEBA(root=opts.root, train=False, transform=transforms.ToTensor(), label=label, Ntest=1000)
		testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
		print 'Data loaders ready.'

		#train SVM
		print 'train SVM with label %s' % (label)
		svm = train_svm(dae, svm, trainLoader, testLoader, exDir=svmDir, lr=opts.svmLR, label=label)

		#test SVM
		## classification test score
		print 'Do classification'
		svm.eval()
		testScores = 0
		noScores = 0
		for i, data in enumerate(testLoader):
			x, y = prep_data(data, useCUDA=svm.useCUDA)
			score = svm_score(svm, y, x=x, dae=dae) 
			testScores+=score * x.size(0)
			noScores+= x.size(0)
		testScores /= noScores

		f = open(join(svmDir, 'multiSVMoutputs.txt'), 'a')
		f.write('\n'+label+': SVM classification (test) score:'+str(testScores.data[0]))
		f.close()
































































