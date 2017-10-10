import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import os
from os.path import join

class DAE(nn.Module):

	def __init__(self, nz, imSize, fSize=2, sigma=0.1):  #sigma is the corruption level
		super(DAE, self).__init__()
		#define layers here

		self.fSize = fSize
		self.nz = nz
		self.imSize = imSize
		self.sigma = sigma

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

	def sample_z(self, noSamples):
		z = torch.randn(noSamples, self.nz)
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
		return self.decode(z)

	def rec_loss(self, rec_x, x, loss='BCE'):
		if loss is 'BCE':
			return F.binary_cross_entropy(rec_x, x, size_average=True)  #not averaged over mini-batch if size_average=FALSE and is averaged if =True 
		elif loss is 'MSE':
			return F.mse_loss(rec_x,x, size_average=True)
		else:
			print 'unknown loss:'+loss

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'dae_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'dae_params')))



# class DIS(nn.Module):




