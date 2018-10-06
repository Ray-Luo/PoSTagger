import torch
import torch.nn as nn

class WordModel():

	@abstractmethod
	def get_char_representation(self):
		pass

	@abstractmethod
	def get_word_representation(self):
		pass

	@abstractmethod
	def get_combined_representation(self):
		pass