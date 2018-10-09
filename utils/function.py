from .NLPDataLoader import *
import pandas as pd

def printParameterSummary(data):
    if data.mode == 'training':
        print("**"*50)
        print("Parameter Summary Start")
        print("                 mode: %s"%(data.mode))
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
        print("      model_save_path: %s"%(data.model_save_path))
        print("       data_save_path: %s"%(data.data_save_path))
        print("                  NER: %s"%(data.NER))
        print("                  GPU: %s"%(data.GPU))
        print("Parameter Summary End")
        print("**"*50)
    else:
        print("**"*50)
        print("Parameter Summary Start")
        print("                 mode: %s"%(data.mode))
        print("           infer_path: %s"%(data.infer_path))
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
        print("      model_save_path: %s"%(data.model_save_path))
        print("       data_save_path: %s"%(data.data_save_path))
        print("     result_save_path: %s"%(data.result_save_path))
        print("                  NER: %s"%(data.NER))
        print("                  GPU: %s"%(data.GPU))
        print("Parameter Summary End")
        print("**"*50)




def getInstances(data, file_path):
    file = pd.read_csv(file_path,header=None,low_memory=False,encoding='utf-8')
    if len(file.columns) > 2:
        column0, column1, column2 = file[file.columns[0]],file[file.columns[1]],file[file.columns[2]]
    else:
        # during inference, create dummy column2
        column0, column1= file[file.columns[0]],file[file.columns[1]]
        column2 = np.zeros((len(column0), 1))
        column2 = [list(data.tag_to_idx.keys())[0] for i in column2]
        column2 = pd.DataFrame(column2,columns=[3])
        column2 = column2[column2.columns[0]]
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

    STRINGS.append(string_tmp)
    TAGS.append(tag_tmp)
    return STRINGS, TAGS

def getDataLoader(file_path, data):
    STRINGS, TAGS = getInstances(data, file_path)
    return get_loader(STRINGS, TAGS, \
                                       data.word_to_idx, data.tag_to_idx,\
                                       data.char_to_idx,\
                                       data.batch_size)


def calculateFScore(true_lists, predict_lists):
    sentence_nb = len(true_lists)
    true_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sentence_nb):
        true_list = true_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(true_list)):
            if true_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(true_list)

        true_matrix = get_matrix(true_list)
        pred_matrix = get_matrix(predict_list)

        right_ner = list(set(true_matrix).intersection(set(pred_matrix)))
        true_full += true_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    true_num = len(true_full)
    predict_num = len(predict_full)

    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if true_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/true_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_score = -1
    else:
        f_score = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("true_num = ", true_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_score

def get_matrix(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)

    return stand_matrix

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def get_instance(idx):
    if idx == 0:
        return 1;
    else:
        return idx;