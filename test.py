import torch
from torch.autograd import Variable

from torchvision import transforms, datasets

from models import DAE, DIS_Z
from dataload import CELEBA
from function import shift_x

from matplotlib import pyplot as plt

import math
import numpy as np

ROOT = '/data'

def test_DAE():
	print 'testing DAE module'
	NZ=100
	FILTER_SIZE=2 #2 for testing
	BATCH_SIZE = 5
	IM_SIZE = 64

	x = Variable(torch.randn(BATCH_SIZE,3,IM_SIZE,IM_SIZE))  #random input
	dae = DAE(imSize=IM_SIZE, fSize=FILTER_SIZE, nz=NZ, sigma=1.0)

	x_corr = dae.corrupt(x)
	z_enc = dae.encode(x)
	z_enc_, x_rec = dae.forward(x)
	z_sample = dae.sample_z(BATCH_SIZE)

	assert x_corr.size() == (BATCH_SIZE, 3, IM_SIZE, IM_SIZE)
	assert z_enc.size() == (BATCH_SIZE, NZ)
	assert z_enc_.size() == (BATCH_SIZE, NZ)
	assert x_rec.size() == (BATCH_SIZE, 3, IM_SIZE, IM_SIZE)
	assert z_sample.size() == (BATCH_SIZE, NZ)

def test_DIS_Z():
	print 'testing DIS_Z module'
	NZ = 100
	BATCH_SIZE = 5

	z = Variable(torch.randn(BATCH_SIZE, NZ))
	dis = DIS_Z(nz=NZ)

	pReal = dis.discriminate(z)
	pReal_ = dis.forward(z)
	lossD = dis.dis_loss(z)
	lossG = dis.gen_loss(z)

	assert pReal.size() == (BATCH_SIZE, 1)
	assert pReal_.size() == (BATCH_SIZE, 1)
	assert not math.isnan(lossD.data[0])
	assert not math.isnan(lossG.data[0])

def test_load_celebA():
	print 'testing data loader'
	BATCH_SIZE = 5
	trainDataset = CELEBA(root=ROOT, train=True, transform=transforms.ToTensor())
	trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)

	(x,y) = iter(trainLoader).next()
	
	assert x.size() == (5, 3, 64, 64)
	assert y.size() == (5, )

def test_shift_x():
	print 'testing shift x'
	BATCH_SIZE = 5
	IM_SIZE = 64

	x = Variable(torch.Tensor(1,1,5,5).fill_(1))  #random input

	DX, DY = 1, 1
	GT=np.zeros((1,1,5,5))
	GT[:,:,1:,1:]=1
	print GT
	xShift = shift_x(x, DX, DY)
	print xShift.data
	assert (xShift.data - torch.Tensor(GT)).abs().mean() < 1e-3


def test_DAE_samples_z():
	print 'testing DAE module'
	NZ=25
	FILTER_SIZE=2 #2 for testing
	BATCH_SIZE = 10000
	IM_SIZE = 64

	x = Variable(torch.randn(BATCH_SIZE,3,IM_SIZE,IM_SIZE))  #random input
	dae = DAE(imSize=IM_SIZE, fSize=FILTER_SIZE, nz=NZ, sigma=1.0, multimodalZ=True)

	z_sample = dae.sample_z(BATCH_SIZE)
	print type(z_sample), z_sample.size()

	z = z_sample.cpu().data.numpy()
	plt.figure()
	plt.hist2d(z[:,0], z[:,1], (100,100))
	plt.title('2D hist for DAE with multimodalZ')
	plt.savefig('2Dhist_z.png')

def test_DIS_samples_z():
	NZ=25
	FILTER_SIZE=2 #2 for testing
	BATCH_SIZE = 10000
	IM_SIZE = 64
	MULTIZ = True

	dae = DAE(nz=NZ, imSize=64, fSize=FILTER_SIZE, sigma=0.1, multimodalZ=MULTIZ) #sigma=level of corruption
	if not dae.multimodalZ:
		prior = dae.norm_prior
	else:
		print 'multi-prior'
		prior = dae.multi_prior
	dis = DIS_Z(nz=NZ, prior=prior)

	z = dis.prior(BATCH_SIZE).numpy()
	print 'z shape:',np.shape(z)
	plt.figure()
	plt.hist2d(z[:,0], z[:,1], (100,100))
	plt.title('2D hist for DAE DIS with multimodalZ')
	plt.savefig('2Dhist_DIS_z.png')



# test_DAE()
# test_DIS_Z()
# test_shift_x()
# test_DAE_samples_z()
test_DIS_samples_z()
# test_load_celebA()



