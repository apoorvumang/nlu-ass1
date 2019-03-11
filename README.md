# Assignment 1 - NLU

Implemented word2vec in python using stochastic gradient descent. Dependencies:

- python3
- numpy
- nltk
- pickle
- tqdm
- scipy
- operator

To **Install** Depedencies run the following command:
```
pip3 install -r requirement.txt
```
## Data Preprocessing
To preprocess data and generate training samples, execute
```
python3 subsample.py
```
## Word2vec Training
To train the **word2vec** model run the following command:
```
python3 train.py --dim 150 --suffix myrun --numneg 15 --epochs 100 --lr 0.01
```

You can type ./train.py -h to get help on arguments that it takes.

## Evaluation

To evaluate a trained model, run 
```
python3 evaluate.py --wfile W_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz --cfile C_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz
```

## K closest words
To get the **10 closest words** of the *test word*, run the following code:
```
python3 findclosest.py --wfile W_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz --cfile C_NEG_20_DIM_300_EPOCHS_25_nonum_full_win3_epoch75.npy.npz --word test_word
```
