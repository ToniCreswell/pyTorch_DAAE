#Interpolations for an iDAAE trained on celebA faces
import sys
sys.path.append('../')

from function import prep_data, make_new_folder, plot_losses, save_input_args, shift_x, plot_norm_losses
from dataload import CELEBA 
from models import IDAE, DIS_Z, LINEAR_SVM

import torch
from torch import optim
from torch.autograd import Variable

from torchvision.utils import make_grid, save_image
from torchvision import transforms, datasets

import argparse

from time import time

import os
from os.path import join

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='/data', type=str) #data/datasets/LabelSwap
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--nz', default=200, type=int)
	parser.add_argument('--fSize', default=64, type=int)  # multiple of filters to use
	parser.add_argument('--load_DAE_from', required=True, type=str)
	parser.add_argument('--alpha', default=1.0, type=float)  # weight on the adversarial cost
	parser.add_argument('--sigma', default=0.1, type=float)  # noise level
	parser.add_argument('--M', default=5, type=int)  # noise level
	parser.add_argument('--comment', type=str)
	parser.add_argument('--Ntest', default=100, type=int)
	parser.add_argument('--gpuNo', required=True, type=int)

	
	return parser.parse_args()


if __name__=='__main__':

	opts = get_args()

	#Create new subfolder for saving results and training params
	exDir = join(opts.load_DAE_from, 'interps')
	try:
		os.mkdir(exDir)
	except:
		print 'already exists'

	print 'Outputs will be saved to:',exDir
	save_input_args(exDir, opts)

	#Load test data
	testDataset = CELEBA(root=opts.root, train=False, transform=transforms.ToTensor())
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print 'Data loaders ready.'

	#Create model
	dae = IDAE(nz=opts.nz, imSize=64, fSize=opts.fSize, sigma=opts.sigma, M=opts.M) #sigma=level of corruption

	if dae.useCUDA:
		torch.cuda.set_device(opts.gpuNo)
		print 'using gpu:', torch.cuda.current_device()
		dae.cuda()

	#Load the iDAAE models
	print 'loading trained iDAAE...'
	dae.load_params(opts.load_DAE_from)

	#Draw latent samples from the prior
	z1 = dae.sample_z(noSamples=10)
	z2 = dae.sample_z(noSamples=10)

	Z_interps = []
	for a in np.linspace(0.0, 1.1, num=11):
		a = Variable(torch.Tensor([a]).cuda())
		Z_interps.append((1-a)*z1 + a * z2)
	Z_interps = torch.cat(Z_interps, dim=0)
	print('interps:', np.shape(Z_interps.data))

	# if gen.useCUDA:
		# Z_interps = Z_interps.cuda()

	x_interps = dae.decode(Z_interps)
	save_image(x_interps.data, join(exDir, 'interps.png'), nrow=10, normalize=True)





