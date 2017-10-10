import torch
from torch.autograd import Variable

from models import DAE

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


test_DAE()