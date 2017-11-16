import torch
from torch.autograd import Variable

import os
from os.path import join

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

def prep_data(data, useCUDA):
	x, y = data
	if useCUDA:
		x = Variable(x.cuda())
		y = Variable(y.cuda()).view(y.size(0),1).type_as(x)
	else:
		x = Variable(x)
		y = Variable(y).view(y.size(0),1).type_as(x)
	return x,y

def make_new_folder(exDir):
	i=1
	while os.path.isdir(join(exDir,'Ex_'+str(i))):
		i+=1

	os.mkdir(join(exDir,'Ex_'+str(i)))
	return join(exDir,'Ex_'+str(i))

def plot_losses(losses, exDir, epochs=1, title='loss'):
	#losses should be a dictionary of losses 
	# e.g. losses = {'loss1':[], 'loss2:'[], 'loss3':[], ... etc.}
	fig1 = plt.figure()
	assert epochs > 0
	for key in losses:
		noPoints = len(losses[key])
		factor = float(noPoints)/epochs
		plt.plot(np.arange(len(losses[key]))/factor,losses[key], label=key)

	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.title(title)
	fig1.savefig(join(exDir, title+'_plt.png'))

def save_input_args(exDir, opts):
	#save the input args to 
	f = open(join(exDir,'opts.txt'),'w')
	saveOpts =''.join(''.join(str(opts).split('(')[1:])\
		.split(')')[:-1])\
		.replace(',','\n')
	f.write(saveOpts)
	f.close()


def shift_x(x, dx, dy):
	xShift = Variable(torch.Tensor(x.size()).fill_(0)).type_as(x)
	non = lambda s: s if s<0 else None
	mom = lambda s: max(0,s)

	xShift[:, :, mom(dy):non(dy), mom(dx):non(dx)] = x[:, :, mom(-dy):non(-dy), mom(-dx):non(-dx)]
	return xShift

def random_occlusion(x, size):  #TODO
	'''sqaure occlusion or WGN'''

	assert (size,size) <= x.size()

def plot_norm_losses(losses, exDir, epochs=1, title='loss'):
	#losses should be a dictionary of losses 
	# e.g. losses = {'loss1':[], 'loss2:'[], 'loss3':[], ... etc.}
	assert epochs > 0
	fig1 = plt.figure()
	for key in losses:
		y = losses[key]
		y -= np.mean(y)
		y /= ( np.std(y) + 1e-6 ) 
		noPoints = len(losses[key])
		factor = float(noPoints)/epochs
		plt.plot(np.arange(len(losses[key]))/factor,losses[key], label=key)
	plt.xlabel('epoch')
	plt.ylabel('normalised loss')
	plt.legend()
	fig1.savefig(join(exDir, 'norm_'+title+'_plt.png'))



