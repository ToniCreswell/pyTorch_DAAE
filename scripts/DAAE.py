#DAAE trained on celebA faces
import sys
sys.path.append('../')

from function import prep_data
from dataload import CELEBA 
from models import DAE, DIS_Z

from torchvision.utils import make_grid, save_image


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='/data/datasets/LabelSwap', type=str)
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

	return parser.parse_args()



if __name__=='__main__':

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
	dae = DAE(nz=opts.nz, imSize=opts.fSize, fSize=opts.fSize, sigma=opts.sigma) #sigma=level of corruption
	dis = DIS_Z(nz=opts.nz)

	if dae.useCUDA:
		dae.cuda()
		dis.cuda()

	#Create optimizers
	optimDAE = optim.Adam(dae.parameters(), lr = opts.lr)
	optimDIS = optim.SGD(dis.parameters(), lr = opts.lr, momentum=0.9)

	#Keeping track of training
	losses = {'enc': [], 'rec': [], 'dis':[], 'test rec':[]}

	#Start training
	for e in range(opts.maxEpochs):

		for i, data in enumerate(trainLoader):

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
			optimDIS.zero_grads()
			disLoss.backward()
			optimDIS.step()

			optimDAE.zero_grads()
			daeLoss.backward()
			optimDAE.step()

			# storing losses for plotting later
			losses['enc'].append(encLoss.mean().data[0])
			losses['rec'].append(recLoss.mean().data[0])
			losses['dis'].append(disLoss.mean().data[0])

			if i%100 == 0:
				print 'enc: %0.5f, rec: %0.5f, dis: %0.5f'

		#### Test
		dae.eval()
		dis.eval()

		#get test outuputs and losses
		xTest, yTest = prep_data(iter(testLoader).next(), useCUDA=dae.useCUDA)
		zTest, recTest = dae.forward(xTest)
		recLossTest = dae.rec_loss(recTest, xTest)

		#Plot losses
		losses['test rec'].append(recLossTest.mean().data[0])
		if e > 0: #only one point for test rec otherwise
			plot_losses(losses, exDir, epochs=e+1)

		#save parameters
		dae.save_params(exDir)
		dis.save_params(exDir)

		#Save images of original and rec
		save_image(xTest.data[0], join(exDir, 'original.png'))
		save_image(recTest.data[0], join(exDir, 'rec.png'))

		#Save samples
		sampleDir = os.mkdir(join(exDir,'epoch_'+str(i)))
		dae.sample_x(opts.M, sampleDir)









