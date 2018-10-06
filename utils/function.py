from .NLPDataLoader import *
import pandas as pd

def printParameterSummary(data):
    print("**"*30)
    print("Parameter Summary Start")
    print("        training_path: %s"%(data.training_path))
    print("      validation_path: %s"%(data.validation_path))
    print("      evaluation_path: %s"%(data.evaluation_path))
    print("   char_embedding_dim: %s"%(data.char_embedding_dim))
    print("   word_embedding_dim: %s"%(data.word_embedding_dim))
    print("      char_hidden_dim: %s"%(data.char_hidden_dim))
    print("      word_hidden_dim: %s"%(data.word_hidden_dim))
    print("                epoch: %s"%(data.epoch))
    print("        learning_rate: %s"%(data.learning_rate))
    print(" pretained_word_embed: %s"%(data.pretrained_word_embed_path))
    print("           char_model: %s"%(data.char_model))
    print("           word_model: %s"%(data.word_model))
    print("           batch_size: %s"%(data.batch_size))
    print("            optimizer: %s"%(data.optimizer))
    print("            save_path: %s"%(data.save_path))
    print("                  GPU: %s"%(data.GPU))
    print("Parameter Summary End")
    print("**"*30)



def getInstances(file_path):
    data = pd.read_csv(file_path,header=None,low_memory=False,encoding='utf-8')
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

    return STRINGS, TAGS

def getDataLoader(file_path, data):
    STRINGS, TAGS = getInstances(file_path)
    return get_loader(STRINGS, TAGS, \
                                       data.word_to_idx, data.tag_to_idx,\
                                       data.char_to_idx,\
                                       data.batch_size)
