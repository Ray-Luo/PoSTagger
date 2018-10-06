from ../utils.pretrained_wordembed import *

class Data:

    def __init__(self, parser):
        self.files = parser.files
        self.char_to_idx = {}
        self.tag_to_idx = {}
        self.word_to_idx = {}
        self.reverse_tag_to_idx = {}
        self.pretrained_wordEmbed_path = parser.pretrained_wordEmbed_path
        

    def buildDictionary(self):
        frames = [pd.read_csv(file,header=None,low_memory=False,encoding='utf-8') for file in self.files]
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
            sentence = iitem[0]
            tags = iitem[1]
            for tag in tags:
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx) + 1
                    self.reverse_tag_to_idx[len(self.tag_to_idx)+1] = tag

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


    def fixDictionary(self):
        self.char_to_idx.close()
        self.tag_to_idx.close()
        self.word_to_idx.close()
        self.reverse_tag_to_idx.close()

    def getPretrainedEmbedding(self):
        self.pretrain_word_emb = build_pretrain_embedding(embedding_path=self.pretrained_wordEmbed_path,\
                                                    word_alphabet = list(self.word_to_idx.keys()))

