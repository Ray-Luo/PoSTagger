from model.Data import Data
from utils.NLPDataLoader import NLPDataLoader
from utils.function import *
import random
from model.SequenceModel import SequenceModel
import torch
import torch.optim as optim

def evaluate(model, instances, data):
    pred_results = []
    true_results = []

    for item in instances:
        with torch.no_grad():
            word_seq, word_seq_lengths = item[0].to(device), item[1].to(device)
            word_seq_recover = item[2].to(device)
            char_seq,char_seq_lengths,char_seq_recover = item[3].to(device),item[4].to(device),item[5].to(device)
            true_tags = item[6].to(device)
            mask = item[7].to(device)
                                            
            batch_size, seq_len = word_seq.size()
            model.zero_grad()

            loss, pred_tag_seq = model.calculateLoss(data, word_seq, word_seq_lengths, \
                char_seq, char_seq_lengths,char_seq_recover, \
                true_tags, mask)

            pred_label, true_label = recover_label(data, pred_tag_seq, true_tags, mask, list(data.tag_to_idx.keys()), word_seq_recover)
            pred_results += pred_label
            true_results += true_label

    accuracy, precision, recall, f_score = calculateFScore(true_results, pred_results)
    return accuracy, precision, recall, f_score


def recover_label(data, pred_variable, true_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            true_variable (batch_size, sent_len): true result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    true_variable = true_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = true_variable.size(0)
    seq_len = true_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    true_tag = true_variable.cpu().data.numpy()
    pred_label = []
    true_label = []
    for idx in range(batch_size):
        pred = [data.reverse_tag_to_idx[get_instance(pred_tag[idx][idy])] for idy in range(seq_len) if mask[idx][idy] != 0]
        true = [data.reverse_tag_to_idx[get_instance(true_tag[idx][idy])] for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(true))
        pred_label.append(pred)
        true_label.append(true)
    return pred_label, true_label

def train(training_instances, validation_instances, evaluation_instances, data):
    print("Training started...")
    model = SequenceModel(data)
    if data.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=data.learning_rate)
    model.to(device)
    best_score = 0
    for epoch in range(data.epoch):
        right_token = 0
        whole_token = 0
        total_loss = 0
        sub_loss = 0
        model.zero_grad()
        for i, item in enumerate(training_instances):
            i += 1
            word_seq, word_seq_lengths = item[0].to(device), item[1].to(device)
            word_seq_recover = item[2].to(device)
            char_seq,char_seq_lengths,char_seq_recover = item[3].to(device),item[4].to(device),item[5].to(device)
            true_tags = item[6].to(device)
            mask = item[7].to(device)

            loss, pred_tag_seq = model.calculateLoss(data, word_seq, word_seq_lengths, \
                char_seq, char_seq_lengths,char_seq_recover, \
                true_tags, mask)
            mask = mask.cpu().data.numpy()
            pred = pred_tag_seq.cpu().data.numpy()
            true = true_tags.cpu().data.numpy()
            overlapped = (pred == true)
            right = np.sum(overlapped * mask)
            whole = mask.sum()
            right_token += right
            whole_token += whole
            sub_loss = loss.item()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()


            instances_nb = i*data.batch_size
            if instances_nb % (data.batch_size*10) == 0:
                print("Instance: {:05d}, sub loss: {:10.10}; sub acc: {:06d}/{:06d} = {:0.4f}".format(instances_nb, sub_loss, right_token, whole_token,(right_token+0.)/whole_token))

        print("Instance: {:05d}, sub loss: {:10.10}; sub acc: {:06d}/{:06d} = {:0.4f}".format(instances_nb, sub_loss, right_token, whole_token,(right_token+0.)/whole_token))

        if epoch % 1 == 0:
            print("Epoch: {:04d}, total loss: {:10.10}; total acc: {:06d}/{:06d} = {:0.4f}".format(epoch+1, total_loss, right_token, whole_token,(right_token+0.)/whole_token))

        if str(total_loss) == 'nan' or total_loss > 1e8:
            print("Error: loss explosion (>1e8) ! Please set proper parameters! Exit....")
            exit(1)

        # validation 
        accuracy, precision, recall, f_score = evaluate(model, validation_instances, data)
        print("Validation --> accuracy: {:0.4f}, precision: {:0.4f}, recall: {:0.4f}, f_score: {:0.4f}".format(accuracy, precision, recall, f_score))

        # evaluation
        accuracy, precision, recall, f_score = evaluate(model, evaluation_instances, data)
        print("Evaluation --> accuracy: {:0.4f}, precision: {:0.4f}, recall: {:0.4f}, f_score: {:0.4f}".format(accuracy, precision, recall, f_score))  

        if data.NER:           
            if best_score < f_score:
                best_score = f_score
                torch.save(model, data.model_save_path)
                print("Better f_score found. Saving the model...")
        else:
            if best_score < accuracy:
                best_score = accuracy
                torch.save(model, data.model_save_path)
                print("Better accuracy found. Saving the model...")
        print("**"*50)

    if data.NER:           
        print("Training finished. Best f_score is {:0.4f}.".format(best_score))
    else:
        print("Training finished. Best accuracy is {:0.4f}.".format(best_score))




seed_num = 123
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


if __name__ == '__main__':
    data = Data()
    train_config_path = './train.config'
    data.readConfig(train_config_path)
    data.buildDictionary()
    data.getPretrainedEmbedding()

    # print parameter summary
    printParameterSummary(data)

    # build dataloaders
    training_instances = getDataLoader(data.training_path, data)
    validation_instances = getDataLoader(data.validation_path, data)
    evaluation_instances = getDataLoader(data.evaluation_path, data)
    data.saveData()

    device = torch.device("cuda:"+data.GPU if torch.cuda.is_available() else "cpu")
    train(training_instances, validation_instances, evaluation_instances, data)



