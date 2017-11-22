1. Run evaluation where a pre-trained SVM exists in the same dir as the DAE

python DAAE.py --evalMode --load_DAE_from=../../Experiments/DAAE/Ex_$EX --commit=eval --alpha=1.0 --sigma=0.0 --root=/data/datasets/LabelSwap/ --loadDAE --M=20 --svmLR=1e-4 --loadSVM

2. Run evaluation where a pre-trained SVM does not exist

python DAAE.py --evalMode --load_DAE_from=../../Experiments/DAAE/Ex_$EX --commit=eval --alpha=1.0 --sigma=0.0 --root=/data/datasets/LabelSwap/ --loadDAE --M=20 --svmLR=1e-4

N.B. in eval mode make sure that sigma is the same as the one used for training.

3. To train a network and SVM
python DAAE.py --commit=xxxxx --alpha=1.0 --sigma=0.0 --root=/data/datasets/LabelSwap/ --M=20 --svmLR=1e-4

4. To continue training a DAAE (not fully working yet, does not load dis and does not save lr if weight decay way used)
NOT IMPLEMENTED COMPLETELY — can just load a DAE net.
