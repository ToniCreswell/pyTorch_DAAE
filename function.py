from torch.autograd import Variable

from torchvision.utils import make_grid, save_image

import os
from os.path import join

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




