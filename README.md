# ProDOMA
ProDOMA is a deep learning model that conducts domain analysis for third-generation sequencing reads. It uses deep neural networks with 3-frame translation encoding and multi-layer convolutional filters to distinguish different protein families. In the experiments on simulated reads of protein coding sequences and real reads from the human genome, our model outperforms profile hidden Markov model based methods and the state-of-the-art deep learning method for protein domain classification. In addition, our model can reject unrelated DNA reads, which is an important function for domain analysis in real transcriptomic and metagenomic data. In summary, ProDOMA is a useful end-to-end protein domain analysis tool for long noisy reads, especially when whole genome assembly is hard.

We are currently cleanning our codes and will release it shortly.

Dependecies
----------

* Python 3 (You may need to modify some script if you want to use Python 2.7)
* [PyTorch](https://pytorch.org/) an open source machine learning framework
* [Biopython](http://biopython.org/) library for Python

References
----------

how to cite this tool:

    Du N., and Sun Y., ProDOMA: improve PROtein DOMAin classification for third-generation sequencing reads using deep learning, submitted to ISMB 2020
