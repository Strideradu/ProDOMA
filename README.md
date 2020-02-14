# ProDOMA
ProDOMA is a deep learning model that conducts domain analysis for third-generation sequencing reads. It uses deep neural networks with 3-frame translation encoding and multi-layer convolutional filters to distinguish different protein families. In the experiments on simulated reads of protein coding sequences and real reads from the human genome, our model outperforms profile hidden Markov model based methods and the state-of-the-art deep learning method for protein domain classification. In addition, our model can reject unrelated DNA reads, which is an important function for domain analysis in real transcriptomic and metagenomic data. In summary, ProDOMA is a useful end-to-end protein domain analysis tool for long noisy reads, especially when whole genome assembly is hard.

We are currently cleanning our codes and will release it shortly.

Usage
----------
```
usage: train.py [-h] [--filter_sizes FILTER_SIZES [FILTER_SIZES ...]]
                [--num_filters NUM_FILTERS] [--num_hidden NUM_HIDDEN]
                [--l2 L2] [--dropout DROPOUT] [--num_classes NUM_CLASSES]
                [--seq_len SEQ_LEN] [--lr LR] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--train_file TRAIN_FILE]
                [--valid_file VALID_FILE] [--test_file TEST_FILE]
                [--ood_file OOD_FILE] [--checkpoint_path CHECKPOINT_PATH]
                [--log_dir LOG_DIR] [--log_interval LOG_INTERVAL]
                [--save_interval SAVE_INTERVAL] [--fine_tuning FINE_TUNING]
                [--fine_tuning_layers FINE_TUNING_LAYERS [FINE_TUNING_LAYERS ...]]
                [--save_prediction SAVE_PREDICTION] [--topk TOPK]
                [--predict_file PREDICT_FILE] [--iter_val ITER_VAL]
                [--model MODEL] [--openset_model OPENSET_MODEL]
                [--filter_topk FILTER_TOPK] [--suffix SUFFIX] [--truth TRUTH]
                [--embed_dim EMBED_DIM] [--conv1_filter CONV1_FILTER]
                [--epsilon EPSILON] [--temperature TEMPERATURE]
                [--repeat REPEAT] [--threshold THRESHOLD] 
                [--apex]

optional arguments:
  -h, --help            show this help message and exit
  --filter_sizes FILTER_SIZES [FILTER_SIZES ...]
                        Space seperated list of motif filter lengths. (ex,
                        --filter_sizes 4 8 12)
  --num_filters NUM_FILTERS
                        number of filters per kernel
  --num_hidden NUM_HIDDEN
                        Number of neurons in hidden layer.
  --l2 L2               (Lambda value / 2) of L2 regularizer on weights
                        connected to last layer (0 to exclude).
  --dropout DROPOUT     Rate for dropout.
  --num_classes NUM_CLASSES
                        Number of classes (families).
  --seq_len SEQ_LEN     Length of input sequences.
  --lr LR               Initial learning rate.
  --epochs EPOCHS       Number of epochs to train.
  --batch_size BATCH_SIZE
                        Batch size. Must divide evenly into the dataset sizes.
  --train_file TRAIN_FILE
                        Directory for input data.
  --valid_file VALID_FILE
                        Directory for input data.
  --test_file TEST_FILE
                        Directory for input data.
  --ood_file OOD_FILE   Directory for input ood data.
  --checkpoint_path CHECKPOINT_PATH
                        Path to write checkpoint file.
  --log_dir LOG_DIR     Directory for log data.
  --log_interval LOG_INTERVAL
                        Interval of steps for logging.
  --save_interval SAVE_INTERVAL
                        Interval of steps for save model.
  --fine_tuning FINE_TUNING
                        If true, weight on last layer will not be restored.
  --fine_tuning_layers FINE_TUNING_LAYERS [FINE_TUNING_LAYERS ...]
                        Which layers should be restored. Default is ["fc2"].
  --save_prediction SAVE_PREDICTION
                        Path to save prediction
  --topk TOPK           Top k prediction for predict
  --predict_file PREDICT_FILE
                        path for predict data.
  --iter_val ITER_VAL   start epoch
  --model MODEL         model name
  --openset_model OPENSET_MODEL
                        model name for openset
  --suffix SUFFIX       suffix of the output folder
  --truth TRUTH         ground truth of the simulated reads
  --embed_dim EMBED_DIM
                        embedding dimension.
  --conv1_filter CONV1_FILTER
                        the filter size for the first convolution of CNNDeep.
  --epsilon EPSILON     epsilon for ODIN.
  --temperature TEMPERATURE
                        temperature for ODIN.
  --repeat REPEAT       repeat for experiments.
  --threshold THRESHOLD
                        path to threshold file.
  --apex
```

Dependecies
----------

* Python 3 (You may need to modify some script if you want to use Python 2.7)
* [PyTorch](https://pytorch.org/) an open source machine learning framework
* [Biopython](http://biopython.org/) library for Python

References
----------

how to cite this tool:

    Du N., and Sun Y., ProDOMA: improve PROtein DOMAin classification for third-generation sequencing reads using deep learning, submitted to ISMB 2020
