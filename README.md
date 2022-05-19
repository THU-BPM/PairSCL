# PairSCL - pair-level supervised contrastive learning

The source code of paper ["PAIR-LEVEL SUPERVISED CONTRASTIVE LEARNING FOR NATURAL LANGUAGE INFERENCE"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746499) published at ICASSP 2022.

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