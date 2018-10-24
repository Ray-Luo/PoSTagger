# PoSTagger
This model is based on word-character level biLSTM and is implemented using PyTorch. It can perform PoS (part of speech) tagging and NER (name entity recognition). For a more detailed background information please visit my [blog post](http://leiluoray.com/2018/10/20/Part-of-Speech-Tagging/).
## Requirement
```
Python 3.6, PyTorch 0.4
```
## Usage
For ***training*** <br>
`python3 train.py`

For ***inferring*** <br>
`python3 predict.py`

## Advantages
* Easy to extend: word-level and char-level learning based on RNN is provided. CNN version can be added by inheriting WordModel and CharModel.
* Fully configurable: all the parameters are defined in a config file. User can also use different pretrained word and char embeddings.
* GPU support: Custom dataloader is defined for fast GPU computation.
* State-of-the-art performance: the model achieved/surpassed the state-of-art performance on several dataset, such as CoNLL2000 and GENIA dataset.


## Data Format
A sample training and testing dataset is provided. The first column is sentence id starting from 0. The second column is the word itself, and the last column is the PoS tagging. A simple pretrained word embedding is also provided.

## Performance on CoNLL2000
|ID| Model |Author(s)|F1-score   
|---|--------- | -------- | --------
|1| Chunks from the Charniak Parser | Hollingshead, Fisher and Roark (2005), Charniak (2000)|94.20%
|2| Bidirectional LSTM-CRF Model |Huang et al. (2015)| 94.46%
|3| Specialized HMM + voting between different representations | Shen and Sarkar (2005)| 95.23%
|4| This model | Lei Luo (2018)|**96.01%** 
