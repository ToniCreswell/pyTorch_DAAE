# Denoising Adversarial Autoencoder

## Prerequisites

pyTorch http://pytorch.org

Other requirements in requirements.txt

Download data from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0

## Prepare the data

Run the jupyter notebook: DataToTenosrCeleba_all.ipynb to get data tensors.


## To run code


### 1a. To train a DAAE network and SVM
python DAAE.py --commit=xxxxx --alpha=1.0 --sigma=0.25 --root=path/to/data/tensor  --M=20 --svmLR=1e-4 --Ntest=1000 --gpuNo=0

### 1b. To train am iDAAE network and SVM
python iDAAE.py --commit=xxxxx --alpha=1.0 --sigma=0.25 --root=path/to/data/tensor  --M=10 --svmLR=1e-4 --Ntest=1000 --gpuNo=0

### 2. Run evaluation where a pre-trained SVM does not exist

python DAAE.py --evalMode --load_DAE_from=../../Experiments/DAAE/Ex_$EX --commit=eval --alpha=1.0 --sigma=0.25 --root=/data/datasets/LabelSwap/ --loadDAE --M=20 --svmLR=1e-4 --Ntest=1000 --gpuNo=0

### 3. Run evaluation where a pre-trained SVM exists in the same dir as the DAE

python DAAE.py --evalMode --load_DAE_from=../../Experiments/DAAE/Ex_$EX --commit=eval --alpha=1.0 --sigma=0.25 --root=/data/datasets/LabelSwap/ --loadDAE --M=20 --svmLR=1e-4 --loadSVM --Ntest=1000 --gpuNo=0



