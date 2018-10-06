from .WordModel import WordModel
from .CharModel import CharModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WordRNN(WordModel):

    def __init__(self, data):
        super(WordModel, self).__init__()
        self.char_model = CharModel(data)
        self.word_embedding_dim = data.word_embedding_dim
        self.word_hidden_dim = data.word_hidden_dim
        self.pretrain_word_emb = data.pretrain_word_emb
        self.word_embeddings = nn.Embedding(data.vocab_size + 1, self.word_embedding_dim)
        self.wordBiLSTM = nn.LSTM(self.word_embedding_dim*2, self.word_hidden_dim, batch_first=True,bidirectional=True)



    def get_char_representation(self, word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover):
        return self.char_model.get_char_representation(word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover)

    def get_word_representation(self,word_seq):
        word_embeds = self.word_embeddings(word_seq)
        word_embeds.weight = nn.Parameter(torch.from_numpy(self.pretrain_emb))

        return word_embeds

    def get_combined_representation(self,word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover):
        char_repre = self.get_char_representation(word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover)
        word_repre = self.get_word_representation(word_seq)

        word_list = [word_repre]
        word_list.append(char_repre)

        combined_repre = torch.cat([word_repre, char_repre], 2)
        combined_repre = torch.cat(word_list, 2)
                                  
        combined_repre = pack_padded_sequence(combined_repre, sentence_seq_lengths, batch_first=True)
        
        return combined_repre


