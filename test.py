import torch
from torch.autograd import Variable

from torchvision import transforms, datasets

from models import DAE, DIS_Z
from dataload import CELEBA

import math

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
	x_rec = dae.forward(x)
	z_sample = dae.sample_z(BATCH_SIZE)

	assert x_corr.size() == (BATCH_SIZE, 3, IM_SIZE, IM_SIZE)
	assert z_enc.size() == (BATCH_SIZE, NZ)
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
	assert y.size() == 5


test_DAE()
test_DIS_Z()
test_load_celebA()



