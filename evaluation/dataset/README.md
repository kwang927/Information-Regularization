# Dataset

## Table of Contents
* [DORIS-MAE](#doris-mae)
* [ArguAna](#arguana)
* [WhatsThatBook](#whatsthatbook)

## DORIS-MAE
DORIS-MAE includes 100 complex research queries in AI, CV, NLP, and ML, split into 40 for training and 60 for testing. Each query is associated with a candidate pool of approximately 100 research paper abstracts, with a fine-grained ranking system. The candidate pools are drawn from a corpus of approximately 360,000 computer science papers.

The details of this dataset are in its [paper](https://arxiv.org/pdf/2310.04678.pdf) and [github](https://github.com/Real-Doris-Mae/Doris-Mae-Dataset). To get the entire dataset, please follow the instructions in the github. 

## ArguAna
ArguAna consists of 8,674 paragraph-length arguments, and a test set of 1,406 argument-counterargument pairs. The task is to retrieve the correct counterargument for each test set argument. We use 5,700 arguments, not part of the test set, for synthetic counterargument generation. Due to the symmetry between arguments and counterarguments, we can use our synthetic data to train for retrieval of counterarguments given arguments.

The details of this dataset can be found in BEIR [paper](https://openreview.net/pdf?id=wCu6T5xFjeJ) and [github](https://github.com/beir-cellar/beir. To get the entire dataset, please follow the instructions in the github. 


## WhatsThatBook
WhatsThatBook contains 14,441 queries from users trying to recall specific books, paired with book titles and descriptions. The task is to match each of the 1,445 test set queries with the correct descriptions. We use 4,000 descriptions, exclusive of the test set, for synthetic query generation.

The details of this dataset can be found in [paper](https://arxiv.org/pdf/2305.15053.pdf). Due to small style issues in the original reported dataset, we reorgainzed this dataset while keeping all the contents, provided here as `wtb_dataset.pickle`.
