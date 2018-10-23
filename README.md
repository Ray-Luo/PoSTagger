# PoSTagger
For a more detailed background information please visit my [blog post]().
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
* Fully configurable: all the parameters are defined in a config file. User can use different pretrained word and char embeddings.
* GPU support: Custom dataloader is defined for fast GPU computation.
* State-of-the-art performance: the model achieved/surpassed the state-of-art performance on several dataset, such as CoNLL2000 and GENIA dataset.


## Data Format
A sample training, validation, and evaluation dataset is provided. The first column is sentence id starting from 0. The second column is the word itself, and the last column is the PoS tagging. A simple pretrained word embedding is also provided.

## Performance on CoNLL2000
|ID| Model |F1-score   
|---|--------- | --------
|1| Chunks from the Charniak Parser | 94.20%
|2| Bidirectional LSTM-CRF Model | 94.46%
|3| Specialized HMM + voting between different representations |  95.23%
|4| This model | **96.01%** 
