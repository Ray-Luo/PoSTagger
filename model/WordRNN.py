from .WordModel import WordModel
from .CharRNN import CharRNN
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WordRNN(WordModel):

    def __init__(self, data):
        super(WordRNN, self).__init__(data)
        if data.char_model == 'rnn':
            self.char_model = CharRNN(data)
            self.word_embedding_dim = data.word_embedding_dim
            self.word_embeddings = nn.Embedding(len(data.word_to_idx)+1, self.word_embedding_dim)
            self.pretrain_word_emb = data.pretrain_word_emb


    def get_char_representation(self, word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover):
        return self.char_model.get_char_representation(word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover)

    def get_word_representation(self, data, word_seq):
        word_embeds = self.word_embeddings(word_seq)
        word_embeds.weight = nn.Parameter(torch.from_numpy(data.pretrain_word_emb))

        return word_embeds

    def get_combined_representation(self,data, word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover):
        char_repre = self.get_char_representation(word_seq, word_seq_lengths, char_seqs, char_seq_lengths,char_seq_recover)
        word_repre = self.get_word_representation(data, word_seq)

        word_list = [word_repre]
        word_list.append(char_repre)

        combined_repre = torch.cat([word_repre, char_repre], 2)
        combined_repre = torch.cat(word_list, 2)
                                  
        combined_repre = pack_padded_sequence(combined_repre, word_seq_lengths, batch_first=True)
        
        return combined_repre


