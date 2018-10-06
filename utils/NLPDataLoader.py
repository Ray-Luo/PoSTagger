import torch
import torch.utils.data as data
import torch.autograd as autograd
import numpy as np

class NLPDataLoader(data.Dataset):
	def __init__(self, source, target, source_word2id, target_word2id, char2id, batch_size):
		self.source = source
		self.target = target
		self.source_word2id = source_word2id
		self.target_word2id = target_word2id
		self.char2id = char2id
		self.batch_size = batch_size

	def __getitem__(self, index):
		sentence, tags = self.source[index], self.target[index]
		return self.preprocess(sentence, tags)

	def __len__(self):
		return len(self.source)

	def preprocess(self, sentence, tags):
		word_idx_list = []
		char_idx_list = []
		max_char = max([len(word) for word in sentence])
		for word in sentence:
			word = word.lower()
	        # get word info
			# if word not in self.source_word2id:
			# 	word_idx = 0
			# else:
			word_idx = self.source_word2id[word]
			word_idx_list.append(word_idx)
	        
	        # get character level info
			word_len = len(word)

			chars = []
			for char in word:
				# if char in self.char2id:
				chars.append(self.char2id[char])

			char_idx_list.append(chars)


		target_tensor_list = [self.target_word2id[tag] for tag in tags]

		return torch.Tensor(word_idx_list),\
				 char_idx_list,\
				 torch.Tensor(target_tensor_list)



def collate_fn(data):

	def mergeWordTag(soruce_seqs,target_seqs):
		word_seq_lengths = torch.tensor([len(seq) for seq in soruce_seqs]).long()
		max_seq_len = max(word_seq_lengths)
		batch_size = len(soruce_seqs)
		#print('word_seq_lengths**************',lengths)
		#seq_lengths, perm_idx = lengths.sort(0, descending=True)
		word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
		tag_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
		# seq_tensor = seq_tensor[perm_idx]
		for i, (word_seq, tag_seq) in enumerate(zip(soruce_seqs,target_seqs)):
			end = word_seq_lengths[i]
			word_seq_tensor[i, :end] = word_seq[:end]
			tag_seq_tensor[i, :end] = tag_seq[:end]

		word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
		word_seq_tensor = word_seq_tensor[word_perm_idx]
		tag_seq_tensor = tag_seq_tensor[word_perm_idx]

		mask = word_seq_tensor.clone()
		mask[word_seq_tensor==0] = 0
		mask[word_seq_tensor!=0] = 1
		# a = mask
		# a = a.cpu().data.numpy()
		# print('***********mask1111**********',mask,a.sum())
		# soruce_seqs, source_lengths, target_seqs, , mask

		return word_seq_tensor, word_seq_lengths, tag_seq_tensor, mask

	def mergeChar(seqs):
		batch_size = len(seqs)
		lengths = torch.tensor([len(seq) for seq in seqs]).long()
		max_seq_len = max(lengths)
		word_seq_lengths, word_perm_idx = lengths.sort(0, descending=True)
		char_max = 0
		char_lengths = []

		#seqs = [seqs[idx] + [[0]] * (max_seq_len-len(seqs[idx])) for idx in range(len(seqs))]

		pad_chars = []
		for idx in range(len(seqs)):
			tmp = []
			for i in range(max_seq_len-len(seqs[idx])):
				tmp.append([0])
			pad_chars.append(seqs[idx] + tmp)

		seqs = pad_chars

		char_lengths = [list(map(len, seq)) for seq in seqs]
		
		char_max = max(map(max, char_lengths))
		
		chars_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, char_max))).long()
		char_lengths = torch.LongTensor(char_lengths)

		for i, (seq, seqlen) in enumerate(zip(seqs, char_lengths)):
			for j, (word, wordlen) in enumerate(zip(seq, seqlen)):
				chars_tensor[i, j, :wordlen] = torch.LongTensor(word)

		#print('chars_tensor',chars_tensor.size(),word_seq_lengths.size())
		chars_tensor = chars_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
		char_lengths = char_lengths[word_perm_idx].view(batch_size*max_seq_len,)

		char_lengths, char_perm_idx = char_lengths.sort(0, descending=True)
		chars_tensor = chars_tensor[char_perm_idx]
		_, char_seq_recover = char_perm_idx.sort(0, descending=False)

		return chars_tensor.squeeze(), char_lengths, char_seq_recover.squeeze()


	data.sort(key=lambda x: len(x[0]), reverse=True)
	soruce_seqs, char_seqs, target_seqs = zip(*data)
	soruce_seqs, source_lengths, target_seqs, mask = mergeWordTag(soruce_seqs,target_seqs)
	char_seqs, char_lengths,char_seq_recover = mergeChar(char_seqs)

	return soruce_seqs,source_lengths,\
			char_seqs,char_lengths,char_seq_recover,\
			target_seqs, mask

def get_loader(source, target, source_word2id, target_word2id, char2id,  batch_size=200):
	dataset = NLPDataLoader(source, target, source_word2id, target_word2id, char2id,  batch_size)

	data_loader = torch.utils.data.DataLoader(dataset=dataset,\
												batch_size=batch_size,
												shuffle=True,
												collate_fn=collate_fn,\
												num_workers=0)
	return data_loader





















































