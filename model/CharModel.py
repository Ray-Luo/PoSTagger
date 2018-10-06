import torch
import torch.nn as nn

class CharModel(nn.Module):
    def __init__(self, data):
        super(CharModel, self).__init__()
        

    @abstractmethod
    def get_char_representation(self):
        pass