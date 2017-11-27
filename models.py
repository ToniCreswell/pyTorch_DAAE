import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from torchvision.utils import make_grid, save_image

import numpy as np

import os
from os.path import join


# class PRIOR(object):
# 	def __init__(self, nz):

# 		self.nz = nz
# 		self.multimodalZ = multimodalZ

# def norm_prior(self, noSamples=25):
# 	z = torch.randn(noSamples, self.nz)
# 	return z

# def multi_prior(self, noSamples=25, mode=None):
# 	#make a 2D sqrt(nz)-by-sqrt(nz) grid of gaussians
# 	num = np.sqrt(self.nz) #no of modes in x and y
# 	STD = 1./ (4*num)
# 	modes = np.arange(-1,1,1./num)
# 	print modes
# 	p = np.random.uniform(0,1,(noSamples*2))

# 	if mode is None:
# 		mu = modes[np.floor(p * num).astype(int)]
# 	else:
# 		mu = modes[np.ones(noSamples, 2) * self.mode]

# 	z = torch.Tensor(mu).view(-1,2) + STD * torch.randn(noSamples, 2)
# 	return z

	#prior = PRIOR(opts.nz, opts.mulitmodealZ, mode=None)
	#priorFn = PRIOR.norm_prior

class DAE(nn.Module):

	def __init__(self, nz, imSize, fSize=2, sigma=0.1, multimodalZ=False):  #sigma is the corruption level
		super(DAE, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		self.sigma = sigma
		self.multimodalZ = multimodalZ

		inSize = imSize / ( 2 ** 4)
		self.inSize = inSize

		if multimodalZ:
			NZ = 2
		else:
			NZ = nz

		self.enc1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.enc5 = nn.Linear((fSize * 8) * inSize * inSize, NZ)

		self.dec1 = nn.Linear(NZ, (fSize * 8) * inSize * inSize)
		self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)
	
		self.useCUDA = torch.cuda.is_available()

	def norm_prior(self, noSamples=25):
		z = torch.randn(noSamples, self.nz)
		return z

	def multi_prior(self, noSamples=25, mode=None):
		#make a 2D sqrt(nz)-by-sqrt(nz) grid of gaussians
		num = np.sqrt(self.nz) #no of modes in x and y
		STD = 1./ (2*num)
		modes = np.arange(-1,1,1./num)
		p = np.random.uniform(0,1,(noSamples*2))

		if mode is None:
			mu = modes[np.floor(p * num).astype(int)]
		else:
			mu = modes[np.ones((noSamples, 2), dtype=int) * int(mode)]

		z = torch.Tensor(mu).view(-1,2) + STD * torch.randn(noSamples, 2)
		return z

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = x.view(x.size(0), -1)
		x = self.enc5(x)
		
		return x

	def corrupt(self, x):
		noise = self.sigma * Variable(torch.randn(x.size())).type_as(x)
		return x + noise

	def sample_z(self, noSamples=25, mode=None):
		if not self.multimodalZ:
			z = self.norm_prior(noSamples=noSamples)
		else:
			z = self.multi_prior(noSamples=noSamples, mode=mode)
		if self.useCUDA:
			return Variable(z.cuda())
		else:
			return Variable(z)

	def decode(self, z):
		#define the decoder here
		z = F.relu(self.dec1(z))
		z = z.view(z.size(0), -1, self.inSize, self.inSize)
		z = F.relu(self.dec2(z))
		z = F.relu(self.dec3(z))
		z = F.relu(self.dec4(z))
		z = F.sigmoid(self.dec5(z))

		return z

	def forward(self, x):
		# the outputs needed for training
		x_corr = self.corrupt(x)
		z = self.encode(x_corr)
		return z, self.decode(z)

	def rec_loss(self, rec_x, x, loss='BCE'):
		if loss == 'BCE':
			return torch.mean(bce(rec_x, x, size_average=True))  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		elif loss == 'MSE':
			return torch.mean(F.mse_loss(rec_x, x, size_average=True))
		else:
			print 'unknown loss:'+loss

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'dae_params'))

	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'dae_params')))

	def sample_x(self, M, exDir, z=None):
		if z == None:
			z = self.sample_z(noSamples=25)
		if not self.multimodalZ:
			x_i = self.decode(z)
			save_image(x_i.data, join(exDir, 'samples0.png'))
			for i in range(M):
				z_i, x_i = self.forward(x_i) #corruption already in there!
				save_image(x_i.data, join(exDir, 'samples'+str(i+1)+'.png'))
		else:
			#show samples from a few modes
			maxModes = min(self.nz, 5)  #show at most 5 modes
			for mode in range(maxModes):
				z = self.sample_z(noSamples=25, mode=mode)
				x_i = self.decode(z)
				save_image(x_i.data, join(exDir, 'mode'+str(mode)+'_samples.png'))
				for i in range(M):
					z_i, x_i = self.forward(x_i) #corruption already in there!
					save_image(x_i.data, join(exDir, 'mode'+str(mode)+'_samples'+str(i+1)+'.png'))


class IDAE(nn.Module):

	def __init__(self, nz, imSize, fSize=2, sigma=0.1, M=20):  #sigma is the corruption level
		super(IDAE, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		self.sigma = sigma
		self.M = M

		inSize = imSize / ( 2 ** 4)
		self.inSize = inSize

		self.enc1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
		self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.enc5 = nn.Linear((fSize * 8) * inSize * inSize, nz)

		self.dec1 = nn.Linear(nz, (fSize * 8) * inSize * inSize)
		self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)
	
		self.useCUDA = torch.cuda.is_available()

	def norm_prior(self, noSamples=25):
		z = torch.randn(noSamples, self.nz)
		return z

	def encode(self, x):
		#define the encoder here return mu(x) and sigma(x)
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = x.view(x.size(0), -1)
		x = self.enc5(x)
		
		return x

	def corrupt(self, x):
		noise = self.sigma * Variable(torch.randn(x.size())).type_as(x)
		return x + noise

	def sample_z(self, noSamples=25):
		z = self.norm_prior(noSamples=noSamples)
		if self.useCUDA:
			return Variable(z.cuda())
		else:
			return Variable(z)

	def decode(self, z):
		#define the decoder here
		z = F.relu(self.dec1(z))
		z = z.view(z.size(0), -1, self.inSize, self.inSize)
		z = F.relu(self.dec2(z))
		z = F.relu(self.dec3(z))
		z = F.relu(self.dec4(z))
		z = F.sigmoid(self.dec5(z))

		return z

	def forward(self, x): #intergration occurs here
		# the outputs needed for training
		x_corr = Variable(torch.Tensor(x.size()).fill_(0)).type_as(x)
		for m in range(self.M):
			x_corr += self.corrupt(x)
		x_corr /= self.M
		z = self.encode(x_corr)
		return z, self.decode(z)

	def rec_loss(self, rec_x, x, loss='BCE'):
		if loss == 'BCE':
			return torch.mean(bce(rec_x, x, size_average=True))  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		elif loss == 'MSE':
			return torch.mean(F.mse_loss(rec_x, x, size_average=True))
		else:
			print 'unknown loss:'+loss

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'dae_params'))

	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'dae_params')))

	def sample_x(self, M, exDir, z=None):
		if z == None:
			z = self.sample_z(noSamples=25)
		
		x = self.decode(z)
		save_image(x.data, join(exDir, 'samples.png'))
		

class DIS_Z(nn.Module):

	'''
	Discriminate between z_real and z_fake vectors
	'''

	def __init__(self, nz, prior):  #cannot use torch.randn anymore cause wen called does not know nz
		super(DIS_Z, self).__init__()

		self.nz = nz
		self.prior = prior

		self.dis1 = nn.Linear(nz, 1000)
		self.dis2 = nn.Linear(1000, 1000)
		self.dis3 = nn.Linear(1000, 1)

	def discriminate(self, z):
		z = F.relu(self.dis1(z))
		z = F.relu(self.dis2(z))
		z = F.sigmoid(self.dis3(z))

		return z

	def forward(self, z):
		return self.discriminate(z)

	def dis_loss(self, z):
		zReal = Variable(self.prior(z.size(0))).type_as(z)
		pReal = self.discriminate(zReal)

		zFake = z.detach()  #detach so grad only goes thru dis
		pFake = self.discriminate(zFake)

		ones = Variable(torch.Tensor(pReal.size()).fill_(1)).type_as(pReal)
		zeros = Variable(torch.Tensor(pFake.size()).fill_(0)).type_as(pFake)

		return 0.5 * torch.mean(bce(pReal, ones) + bce(pFake, zeros))

	def gen_loss(self, z):
		# n.b. z is not detached so it will update the models it has passed thru
		pFake = self.discriminate(z)
		ones = Variable(torch.Tensor(pFake.size()).fill_(1)).type_as(pFake)
		return torch.mean(bce(pFake, ones))

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'dis_z_params'))

	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'dis_z_params')))

	def plot_encz(self, exDir):  #TODO
		'''
		plot the encoded z samples
		'''
		print 'TO DO'


class LINEAR_SVM(nn.Module): 
	"""Support Vector Machine"""

	def __init__(self, nz, c):
		super(LINEAR_SVM, self).__init__()
		self.nz = nz
		self.c = c
		self.fc = nn.Linear(nz, 1)
		self.useCUDA = torch.cuda.is_available()
		self.thresh = 0.5

	def forward(self, x):
		h = self.fc(x)
		return h

	def loss(self, output, y):
		# assert (y.min() != 0) #want y in [-1,1]
		loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
		loss += self.c * torch.mean(self.fc.weight**2)  # l2 penalty
		return loss

	def binary_class_score(self, output, target, thresh=0):
		predLabel = torch.gt(output, thresh)
		classScoreTest = torch.eq(predLabel, target.type_as(predLabel))
		return  classScoreTest.float().sum()/target.size(0)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'linearSVM_params'))

	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'linearSVM_params')))




