#DAAE trained on celebA faces
import sys
sys.path.append('../')

from function import prep_data, make_new_folder, plot_losses, save_input_args, shift_x, plot_norm_losses
from dataload import CELEBA 
from models import IDAE, DIS_Z, LINEAR_SVM

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

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='/data', type=str) #data/datasets/LabelSwap
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--maxEpochs', default=10, type=int)
	parser.add_argument('--nz', default=200, type=int)
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--fSize', default=64, type=int)  # multiple of filters to use
	parser.add_argument('--outDir', default='../../Experiments/iDAAE1000/', type=str)
	parser.add_argument('--commit', required=True, type=str)
	parser.add_argument('--alpha', default=1.0, type=float)  # weight on the adversarial cost
	parser.add_argument('--sigma', default=0.1, type=float)  # noise level
	parser.add_argument('--M', default=5, type=int)  #number of sampling iterations
	parser.add_argument('--loss', default='BCE', type=str) #'BCE' or 'MSE' currently supported
	parser.add_argument('--loadDAE', action='store_true')
	parser.add_argument('--loadSVM', action='store_true')
	parser.add_argument('--load_DAE_from', default=None, type=str)
	parser.add_argument('--evalMode', action='store_true')
	parser.add_argument('--comment', type=str)
	parser.add_argument('--momentum', default=0.9, type=float) 
	parser.add_argument('--c', type=float, default=0.01) #for training the linearSVM for eval
	parser.add_argument('--svmLR', type=float, default=1e-4)
	parser.add_argument('--Ntest', default=100, type=int)
	parser.add_argument('--gpuNo', required=True, type=int)

	
	return parser.parse_args()

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

def eval_mode(dae, exDir, M, testLoader, svm=None):
	f = open(join(exDir, 'outputs.txt'), 'w')
	dae.eval()

	## reconstruction error ##
	recError = []
	print 'calculating reconstruction error...'
	for i, data in enumerate(testLoader):
		x, y = prep_data(data, useCUDA=dae.useCUDA)
		zTest = dae.encode(x)
		recTest = dae.decode(zTest)
		recError.append(dae.rec_loss(recTest, x).data[0])
	meanRecError = np.mean(recError)
	f.write('mean reconstruction error (non-corrupted): %0.5f' % (meanRecError))


	#eval samples ##TODO

	## representation robustness (shift) ##
	print 'performing robustness plot...'
	maxShift = x.size(2)//2
	step = 4
	axis = range(-maxShift, maxShift, step)
	robustnessMap = torch.Tensor(maxShift*2//step, maxShift*2//step).fill_(0)
	classMap = torch.Tensor(maxShift*2//step, maxShift*2//step).fill_(0)
	x, y = prep_data(iter(testLoader).next(), useCUDA=dae.useCUDA)  #take a batch of samples
	allShifts=[]
	enc00 = dae.encode(x)
	for j, dx in enumerate(axis):
		for i, dy in enumerate(axis):
			xShift = shift_x(x, dy, dx)
			encDxDy = dae.encode(xShift)
			# diff = [(torch.dot(encDxDy[k], enc00[k])/ (torch.norm(encDxDy[k])*torch.norm(enc00[k]))).data[0] for k in range(encDxDy.size(0))]
			diff = [torch.dot(encDxDy[k], enc00[k]).data[0]/ ((torch.norm(encDxDy[k])*torch.norm(enc00[k])).data[0] + 1e-6) for k in range(encDxDy.size(0))]
			robustnessMap[j,i] = np.mean(diff)
			classMap[j,i] = svm_score(svm, y, enc=encDxDy).data[0]

			allShifts.append(xShift[0].cpu().data.numpy())

	print 'saving images...'
	print type(allShifts), np.shape(allShifts)
	save_image(torch.Tensor(np.asarray(allShifts)), join(exDir,'shiftImages.png'), nrow=16)
	print robustnessMap

	print 'save maps as numpy array...'
	np.save(join(exDir, 'classMap.npy'), classMap.numpy())
	np.save(join(exDir, 'shiftMap.npy'), robustnessMap.numpy())

	# plot shift robustenss map
	fig0 = plt.figure()
	print robustnessMap.min(), robustnessMap.max(), robustnessMap.size()
	f.write('\nrobustness min: %0.5f, max: %0.5f' % (robustnessMap.min(), robustnessMap.max()))
	plt.imshow(robustnessMap.numpy(), extent=[-maxShift, maxShift, -maxShift, maxShift], vmin=0.0, vmax=1.0)
	plt.xlabel('DX')
	plt.ylabel('DY')
	plt.title('Robustness to shifts in x and y')
	plt.colorbar()
	plt.savefig(join(exDir, 'ShiftRobustness.png'))

	#Plot shift robusteness classification map
	fig1 = plt.figure()
	f.write('\nshift robustenss accuracy min: %0.5f, max: %0.5f' % (classMap.min(), classMap.max()))
	f.write('\nAccuray Volume (sum of elements in accuracy shift map): %0.5f' % (np.clip(classMap.numpy(), 0.5, 1).sum()))
	plt.imshow(classMap.numpy(), extent=[-maxShift, maxShift, -maxShift, maxShift], vmin=0.5, vmax=1.0)
	plt.xlabel('DX')
	plt.ylabel('DY')
	plt.title('Classiciation Robustness to shifts in x and y')
	plt.colorbar()
	plt.savefig(join(exDir, 'ClassificationShiftRobustness.png'))

	## Compare histograms for enc, z_samples and encCorr
	fig2 = plt.figure()
	nEnc, bEnc, _ = plt.hist(enc00.cpu().data.numpy().flatten(), 100, normed=True)
	xcorr = dae.corrupt(x)
	encCorr = dae.encode(xcorr)
	nEncCorr, bEncCorr, _ = plt.hist(encCorr.cpu().data.numpy().flatten(), 100, normed=True)
	nNorm, bNorm, _ = plt.hist(dae.sample_z(10000).cpu().data.numpy().flatten(), 100, normed=True)
	fig3 = plt.figure()
	plt.plot(bEnc[1:], nEnc, label='encoding')
	plt.plot(bEncCorr[1:], nEncCorr, label='corrupted encoding')
	plt.plot(bNorm[1:], nNorm, label='Normal')
	plt.title('Comparing Encodings')
	plt.xlabel('Value')
	plt.ylabel('pdf')
	plt.legend()
	plt.savefig(join(exDir, 'HisEnc.png'))

	#save all histograms:
	np.save(join(exDir, 'HistEnc.npy'), [nEnc,bEnc])
	np.save(join(exDir), 'HistEncCorr.npy', [nEncCorr, bEncCorr])
	np.save(join(exDir), 'prior.npy', [nNorm, bNorm])

	#sampling
	print 'sampling...'
	sampleDir = join(exDir,'FinalSamples')
	try:
		os.mkdir(sampleDir)
	except OSError: print 'file alread exists'
	dae.sample_x(opts.M, sampleDir)

	## classification test score
	if svm is not None:
		'Do classification'
		testScore = 0
		for i, data in enumerate(testLoader):
			x, y = prep_data(data, useCUDA=svm.useCUDA)
			score = svm_score(svm, y, x=x, dae=dae) 
			testScore+=score
	testScore /= (i+1)
	f.write('\nSVM classification (test) score:'+str(testScore.data[0]))
	f.close()




def train_svm(dae, svm, trainLoader, testLoader, exDir, lr):
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

	f = open(join(exDir, 'svmOpts.txt'), 'w')
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
			plot_losses(svmLoss, exDir=exDir, epochs=epoch+1, title='SVM_loss')

		#Do classification
		testScore = svm.binary_class_score(testOutputs, yTest) #has threshold as zero for testOutputs in [-1,1]
		trainScore = svm.binary_class_score(output, y) #has threshold as zero for output in [-1,1]
		f = open(join(exDir, 'svm.txt'), 'w')
		f.write('trainScore: %f \ntestScore: %f ' \
		 % (trainScore.mean().data[0], testScore.mean().data[0]))
		f.close()
 	
	return svm


if __name__=='__main__':

	opts = get_args()

	#Load data
	print 'Prepare data loaders...'
	transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
	trainDataset = CELEBA(root=opts.root, train=True, transform=transforms.ToTensor())
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opts.batchSize, shuffle=True)

	testDataset = CELEBA(root=opts.root, train=False, transform=transforms.ToTensor())
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print 'Data loaders ready.'

	#Create model
	dae = IDAE(nz=opts.nz, imSize=64, fSize=opts.fSize, sigma=opts.sigma, M=opts.M) #sigma=level of corruption
	dis = DIS_Z(nz=opts.nz, prior=dae.norm_prior)
	svm = LINEAR_SVM(nz=opts.nz, c=opts.c) #model


	if dae.useCUDA:
		torch.cuda.set_device(opts.gpuNo)
		print 'using gpu:', torch.cuda.current_device()
		dae.cuda()
		dis.cuda()
		svm.cuda()


	if opts.loadDAE:  #should load DAE if in eval mode
		print 'loading DAE...'
		dae.load_params(opts.load_DAE_from)

	if opts.loadSVM:
			svm.load_params(opts.load_DAE_from) #use SVM @ same location as DAE [may not be one there]
	
	if opts.evalMode & (not opts.loadSVM):  #to train an SVM for eval
			svm = train_svm(dae=dae, svm=svm, trainLoader=trainLoader, testLoader=testLoader, exDir=opts.load_DAE_from, lr=opts.svmLR)

	if opts.evalMode:
		assert opts.loadDAE == True
		eval_mode(dae, opts.load_DAE_from, opts.M, testLoader, svm=svm)
		exit()
	else:
		#Create a folder for this experiment
		exDir = make_new_folder(opts.outDir)
		print 'Outputs will be saved to:',exDir
		save_input_args(exDir, opts)  #save training opts

	#Create optimizers
	optimDAE = optim.RMSprop(dae.parameters(), lr = opts.lr)
	optimDIS = optim.RMSprop(dis.parameters(), lr = opts.lr, momentum=opts.momentum)

	#Keeping track of training
	losses = {'enc': [], 'rec': [], 'dis':[], 'test rec':[]}

	#Start training
	for e in range(opts.maxEpochs):

		epochEncLoss=0
		epochRecLoss=0
		epochDisLoss=0

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
			epochEncLoss+=encLoss.data[0]
			epochRecLoss+=recLoss.data[0]
			epochDisLoss+=disLoss.data[0]

			if i%100 == 0:
				print '[%d, %d] enc: %0.5f, rec: %0.5f, dis: %0.5f, time: %0.3f' % (e, i, encLoss.data[0], recLoss.data[0], disLoss.data[0], time() - T)

		# storing losses for plotting later
		losses['enc'].append(epochEncLoss/i)
		losses['rec'].append(epochRecLoss/i)
		losses['dis'].append(epochDisLoss/i)

		#### Test
		dae.eval()
		dis.eval()

		#get test outuputs and losses
		xTest, yTest = prep_data(iter(testLoader).next(), useCUDA=dae.useCUDA)
		zTest, recTest = dae.forward(xTest)  #N.B. corruption in here
		recLossTest = dae.rec_loss(recTest, xTest)

		#Plot losses
		losses['test rec'].append(recLossTest.data[0])
		if e > 0: #only one point for test rec otherwise
			plot_losses(losses, exDir, epochs=e+1)
			plot_norm_losses(losses, exDir)

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

	if not opts.evalMode:
		eval_mode(dae=dae, exDir=exDir, M=20, testLoader=testLoader, svm=svm)
		svm = train_svm(dae=dae, svm=svm, trainLoader=trainLoader, testLoader=testLoader, exDir=exDir, lr=opts.svmLR)


	#Train a linear-SVM classifier on the enocdings











