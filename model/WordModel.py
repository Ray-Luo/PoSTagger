import torch
import torch.nn as nn
from abc import abstractmethod

class WordModel(nn.Module):
    def __init__(self, data):
        super(WordModel, self).__init__()
        

    @abstractmethod
    def get_char_representation(self):
        pass

    @abstractmethod
    def get_word_representation(self):
        pass

    @abstractmethod
    def get_combined_representation(self):
        pass