#DAAE trained on celebA faces
import sys
sys.path.append('../')

from function import prep_data, make_new_folder, plot_losses, save_input_args, shift_x
from dataload import CELEBA 
from models import DAE, DIS_Z

import torch
from torch import optim

from torchvision.utils import make_grid, save_image
from torchvision import transforms, datasets

import argparse

from time import time

import os
from os.path import join

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='/data', type=str)
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--maxEpochs', default=10, type=int)
	parser.add_argument('--nz', default=100, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--fSize', default=64, type=int)  # multiple of filters to use
	parser.add_argument('--outDir', default='../../Experiments/DAAE/', type=str)
	parser.add_argument('--commit', required=True, type=str)
	parser.add_argument('--alpha', default=1, type=float)  # weight on the adversarial cost
	parser.add_argument('--sigma', default=0.1, type=float)  # noise level
	parser.add_argument('--M', default=5, type=int)  #number of sampling iterations
	parser.add_argument('--loss', default='BCE', type=str) #'BCE' or 'MSE' currently supported
	parser.add_argument('--loadDAE', action='store_true')
	parser.add_argument('--load_DAE_from', default=None, type=str)
	parser.add_argument('--evalMode', action='store_true')

	return parser.parse_args()


def eval_mode(dae, exDir, M, testLoader):
	f = open(join(exDir, 'outputs.txt'), 'w')
	dae.eval()
	#reconstruction error and t-SNE
	recError = []
	for i, data in enumerate(testLoader):
		x, y = prep_data(data, useCUDA=dae.useCUDA)
		zTest, recTest = dae.forward(x)
		recError.append(dae.rec_loss(recTest, x).data[0])
	meanRecError = torch.mean(recError)
	f.write('mean reconstruction error: %0.5f' % (meanRecError))

	#sampling
	for m in range(M):
		sampleDir = join(exDir,'FinalSamples')
		os.mkdir(sampleDir)
		dae.sample_x(opts.M, sampleDir)

	#eval samples ##TODO

	#representation robustness (shift)
	maxShift = x.size(2)//2
	robustnessMap = torch.Tensor(maxShift, maxShift).fill_(0)
	x,y = prep_data(iter(testLoader).next())  #take a batch of samples
	enc00 = dae.encode(x)
	for dx in range(-maxShift, maxShift):
		for dy in range(-maxShift, maxShift):
			xShift = shift_x(x)
			encDxDy = dae.encode(x)
			diff = [torch.dot(encDxDy[i], enc00[i]) for i in range(encDxDy.size(0))]
			robustnessMap[dx,dy] = torch.mean(diff)

	fig1 = plt.figure()
	plt.imshow(robustnessMap, extent=[-maxShift, maxShift, -maxShift, maxShift])
	plt.colorbar()
	plt.savefig(join(exDir, 'shiftRobustness.png'))

	#classification



if __name__=='__main__':

	opts = get_args()

	#Create a folder for this experiment
	exDir = make_new_folder(opts.outDir)
	print 'Outputs will be saved to:',exDir
	save_input_args(exDir, opts)  #save training opts

	#Load data
	print 'Prepare data loaders...'
	transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
	trainDataset = CELEBA(root=opts.root, train=True, transform=transforms.ToTensor())
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

	testDataset = CELEBA(root=opts.root, train=False, transform=transforms.ToTensor())
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print 'Data loaders ready.'

	#Create model
	dae = DAE(nz=opts.nz, imSize=64, fSize=opts.fSize, sigma=opts.sigma) #sigma=level of corruption
	dis = DIS_Z(nz=opts.nz)

	if dae.useCUDA:
		dae.cuda()
		dis.cuda()

	if opts.loadDAE:
		dae.load_params(opts.load_DAE_from)
	if evalMode:
		eval_mode(dae, opts.load_DAE_from, opts.M, testLoader)
		opts.maxEpochs = 0

	#Create optimizers
	optimDAE = optim.Adam(dae.parameters(), lr = opts.lr)
	optimDIS = optim.SGD(dis.parameters(), lr = opts.lr, momentum=0.9)

	#Keeping track of training
	losses = {'enc': [], 'rec': [], 'dis':[], 'test rec':[]}

	#Start training
	for e in range(opts.maxEpochs):

		for i, data in enumerate(trainLoader):

			T = time()

			dae.train()
			dis.train()

			x, y = prep_data(data, useCUDA=dae.useCUDA)

			# get outputs
			zFake, xRec = dae.forward(x)

			# clac losses
			recLoss = dae.rec_loss(xRec, x, loss=opts.loss)  #loss='BCE' or 'MSE'
			encLoss = dis.gen_loss(zFake)
			disLoss = dis.dis_loss(zFake)

			daeLoss = recLoss + opts.alpha * encLoss

			#do updates
			optimDIS.zero_grad()
			disLoss.backward()
			optimDIS.step()

			optimDAE.zero_grad()
			daeLoss.backward()
			optimDAE.step()

			# storing losses for plotting later
			losses['enc'].append(encLoss.data[0])
			losses['rec'].append(recLoss.data[0])
			losses['dis'].append(disLoss.data[0])

			if i%100 == 0:
				print '[%d, %d] enc: %0.5f, rec: %0.5f, dis: %0.5f, time: %0.3f' % (e, i, encLoss.data[0], recLoss.data[0], disLoss.data[0], time() - T)

		#### Test
		dae.eval()
		dis.eval()

		#get test outuputs and losses
		xTest, yTest = prep_data(iter(testLoader).next(), useCUDA=dae.useCUDA)
		zTest, recTest = dae.forward(xTest)
		recLossTest = dae.rec_loss(recTest, xTest)

		#Plot losses
		losses['test rec'].append(recLossTest.data[0])
		if e > 0: #only one point for test rec otherwise
			plot_losses(losses, exDir, epochs=e+1)

		#save parameters
		dae.save_params(exDir)
		dis.save_params(exDir)

		#Save images of original and rec
		save_image(xTest.data, join(exDir, 'original.png'))
		save_image(recTest.data, join(exDir, 'rec.png'))

		#Save samples
		sampleDir = join(exDir,'epoch_'+str(e))
		os.mkdir(sampleDir)
		print 'sample dir:', sampleDir
		dae.sample_x(opts.M, sampleDir)









