import torch
import torch.nn as nn
import torch.nn.functional as F
from .WordRNN import WordRNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SequenceModel(nn.Module):

    def __init__(self, data):
        super(SequenceModel, self).__init__()
        if data.word_model == 'rnn':
            self.word_model = WordRNN(data)
            self.wordBiLSTM = nn.LSTM(data.word_embedding_dim+2*data.char_hidden_dim, data.word_hidden_dim, batch_first=True,bidirectional=True)
            self.hidden2tag = nn.Linear(data.word_hidden_dim*2, len(data.tag_to_idx)+1)

        

    def calculateLoss(self, data, word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover, true_tags, mask):
        batch_size, seq_len = word_seq.size()

        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        combined_repre = self.word_model.get_combined_representation(data, word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover)
        
        word_hidden = None
        self.wordBiLSTM.flatten_parameters()
        word_lstm_out, word_hidden = self.wordBiLSTM(combined_repre, word_hidden)
        word_lstm_out, _ = pad_packed_sequence(word_lstm_out, batch_first=True)
        word_lstm_out = word_lstm_out.contiguous()
        tag_space = self.hidden2tag(word_lstm_out.view(-1, word_lstm_out.shape[2]))
        tag_space = tag_space.view(batch_size * word_seq_lengths.max(), -1)
        tag_scores = F.log_softmax(tag_space, dim=1)

        batch_size, seq_len = word_seq.size()
        loss = loss_function(tag_scores, true_tags.view(batch_size * seq_len))
        _, pred_tag_seq = torch.max(tag_scores, 1)
        pred_tag_seq = pred_tag_seq.view(batch_size, seq_len)

        return loss, pred_tag_seq




