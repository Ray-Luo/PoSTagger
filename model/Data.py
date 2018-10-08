from utils.pretrained_wordembed import *
import pandas as pd

class Data:

    def __init__(self):
        self.char_to_idx = {}
        self.tag_to_idx = {}
        self.word_to_idx = {}
        self.reverse_tag_to_idx = {}
        self.training_path = None
        self.validation_path = None
        self.evaluation_path = None
        self.char_embedding_dim = None
        self.word_embedding_dim = None
        self.char_hidden_dim = None
        self.word_hidden_dim = None
        self.param_name = None
        self.learning_rate = None
        self.pretrained_word_embed_path = None
        self.pretrain_word_emb = None
        self.char_model = None
        self.word_model = None
        self.batch_size = None
        self.GPU = None
        self.epoch = None
        self.optimizer = None
        self.save_path = None

    def readConfig(self,file_path):
        lines = open(file_path, 'r').readlines()
        for line in lines:
            if len(line) == 1 or line[0] == '#':
                continue
            param_name, value =  line.replace(' ', '').split('=')[0], line.replace(' ', '').split('=')[1].replace('\n','')

            if str(param_name) == "training_path":
                self.training_path = value

            elif param_name == 'validation_path':
                self.validation_path = value

            elif param_name == 'evaluation_path':
                self.evaluation_path = value

            elif param_name == 'char_embedding_dim':
                self.char_embedding_dim = int(value)

            elif param_name == 'word_embedding_dim':
                self.word_embedding_dim = int(value)

            elif param_name == 'char_hidden_dim':
                self.char_hidden_dim = int(value)

            elif param_name == 'word_hidden_dim':
                self.word_hidden_dim = int(value)

            elif param_name == 'epoch':
                self.epoch = int(value)

            elif param_name == 'learning_rate':
                self.learning_rate = float(value)

            elif param_name == 'pretained_word_embed_path':
                self.pretrained_word_embed_path = value

            elif param_name == 'char_model':
                self.char_model = value

            elif param_name == 'word_model':
                self.word_model = value

            elif param_name == 'batch_size':
                self.batch_size = int(value)

            elif param_name == 'GPU':
                self.GPU = value

            elif param_name == 'optimizer':
                self.optimizer = value

            elif param_name == 'save_path':
                self.save_path = value


        

    def buildDictionary(self):
        files = [self.training_path, self.evaluation_path, self.validation_path]
        frames = [pd.read_csv(file,header=None,low_memory=False,encoding='utf-8') for file in files]
        data = pd.concat(frames)
        column0, column1, column2 = data[data.columns[0]],data[data.columns[1]],data[data.columns[2]]
        preIdx = '0'
        preToken = ''
        STRINGS = []
        TAGS = []
        string_tmp = []
        tag_tmp = []
        for idx, token, tag in zip(column0, column1, column2):
            idx = str(idx)
            token = str(token)
            tag = str(tag)
            if preIdx != idx:
                STRINGS.append(string_tmp)
                TAGS.append(tag_tmp)
                string_tmp = []
                tag_tmp = []

            string_tmp.append(token)
            tag_tmp.append(tag)
            preIdx = idx


        for item in zip(STRINGS, TAGS):
            sentence = item[0]
            tags = item[1]
            for tag in tags:
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx) + 1
                    self.reverse_tag_to_idx[len(self.reverse_tag_to_idx)+1] = tag

            sen_list = sentence
            for word in sen_list:
                word = word.lower()
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)+1
                for char in word:
                    if char not in self.char_to_idx:
                        self.char_to_idx[char] = len(self.char_to_idx)+1

        self.word_to_idx["WORD_PAD"] = 0
        self.char_to_idx["CHAR_PAD"] = 0


    def getPretrainedEmbedding(self):
        self.pretrain_word_emb = build_pretrain_embedding(embedding_path=self.pretrained_word_embed_path,\
                                                    word_alphabet = list(self.word_to_idx.keys()))

