# PairSCL - pair-level supervised contrastive learning

The source code of paper ["PAIR-LEVEL SUPERVISED CONTRASTIVE LEARNING FOR NATURAL LANGUAGE INFERENCE"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746499) at ICASSP 2022.

## Environments
pytorch=1.8.1, transformers=4.2.1

## Fetch the data to train and test the model
```
fetch_data.py [-h] [--dataset_url DATASET_URL]
              [--target_dir TARGET_DIR]
```

## Preprocess the data
```
preprocess_*.py [-h] [--config CONFIG]
```


## Train the encoder
```
python main_supcon.py  --epoch EPOCH --batch_size BatchSize --dataset Dataset --dist-url tcp://localhost:10001 --multiprocessing-distributed --world-size 1 --rank 0 
```

## Train the classifier
```
python main_validate.py --dataset Dataset --ckpt pathToModel --dist-url tcp://localhost:10001 --multiprocessing-distributed --world-size 1 --rank 0
```

## Test the model
```
python main_test.py --dataset Dataset --gpu GPU --ckpt_bert pathToEncoder --ckpt_classifier pathToClassifier
```

# Reference
If the code is used in your research, hope you can cite our paper as follows:
```
@INPROCEEDINGS{9746499,
  author={Li, Shu’ang and Hu, Xuming and Lin, Li and Wen, Lijie},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Pair-Level Supervised Contrastive Learning for Natural Language Inference}, 
  year={2022},
  volume={},
  number={},
  pages={8237-8241},
  doi={10.1109/ICASSP43922.2022.9746499}}
  ```
