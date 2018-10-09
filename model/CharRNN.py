from .CharModel import CharModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharRNN(CharModel):

    def __init__(self, data):
        super(CharRNN, self).__init__(data)

        self.char_embedding_dim = data.char_embedding_dim
        self.char_hidden_dim = data.char_hidden_dim
        self.char_embedding = nn.Embedding(len(data.char_to_idx) + 1, self.char_embedding_dim)
        self.charBiLSTM = nn.LSTM(self.char_embedding_dim, self.char_hidden_dim, batch_first=True,bidirectional=True)


    def get_char_representation(self, word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover):
        batch_size, seq_len = word_seq.size()
        char_embeds = self.char_embedding(char_seqs)
        char_embeds = pack_padded_sequence(char_embeds, char_seq_lengths, batch_first=True)
        char_hidden = None
        self.charBiLSTM.flatten_parameters()
        char_lstm_out, char_hidden = self.charBiLSTM(char_embeds, char_hidden)
        char_lstm_out, _ = pad_packed_sequence(char_lstm_out)

        char_repre = char_hidden[0].transpose(1,0).contiguous().view(batch_size*seq_len,-1)

        char_repre = char_repre[char_seq_recover]
        char_repre = char_repre.view(batch_size,seq_len,-1)

        return char_repre



