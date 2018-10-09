from model.Data import Data
from utils.NLPDataLoader import NLPDataLoader
from utils.function import *
import random
from model.SequenceModel import SequenceModel
import torch
import torch.optim as optim


def predict(model, instances, data):
    pred_results = []

    for item in instances:
        with torch.no_grad():
            word_seq, word_seq_lengths = item[0].to(device), item[1].to(device)
            word_seq_recover = item[2].to(device)
            char_seq,char_seq_lengths,char_seq_recover = item[3].to(device),item[4].to(device),item[5].to(device)
            true_tags = item[6].to(device)
            mask = item[7].to(device)
                                            
            batch_size, seq_len = word_seq.size()
            model.zero_grad()

            _, pred_tag_seq = model.calculateLoss(data, word_seq, word_seq_lengths, \
                char_seq, char_seq_lengths,char_seq_recover, \
                true_tags, mask)

            pred_label = recover_label(data, pred_tag_seq, mask, word_seq_recover)
            pred_results += pred_label

    pred_results = [item for sublist in pred_results for item in sublist]
    pred_results = pd.DataFrame(pred_results,columns=['predicted_labels'])
    return pred_results


def recover_label(data, pred_variable, mask_variable, word_recover):

    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    pred_label = []
    for idx in range(batch_size):
        pred = [data.reverse_tag_to_idx[get_instance(pred_tag[idx][idy])] for idy in range(seq_len) if mask[idx][idy] != 0]
        pred_label.append(pred)
    return pred_label


seed_num = 123
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    data = Data()
    data.load('./data/PoSTagger.data')
    predict_config_path = './predict.config'
    data.readConfig(predict_config_path)
    printParameterSummary(data)

    predict_instances = getDataLoader(data.infer_path, data)

    device = torch.device("cuda:"+data.GPU if torch.cuda.is_available() else "cpu")

    model = SequenceModel(data)
    model = torch.load(data.model_save_path)
    model.eval()

    words = pd.read_csv(data.infer_path,header=None,low_memory=False,encoding='utf-8')
    predicted_labels = predict(model, predict_instances, data)
    pred_results = pd.concat([words, predicted_labels],axis=1)
    pred_results.to_csv(data.result_save_path,header=False,index=False)
    print('Results saved at {}.'.format(data.result_save_path))