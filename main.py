from model.Data import Data
from utils.NLPDataLoader import NLPDataLoader
from utils.function import *
from model.SequenceModel import SequenceModel
import torch
import torch.optim as optim


def train(training_instances, data):
    print("Training started...")
    torch.manual_seed(1)
    model = SequenceModel(data)
    if data.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=data.learning_rate)
    device = torch.device("cuda:"+data.GPU if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(data.epoch):
        right_token = 0
        whole_token = 0
        total_loss = 0
        sub_loss = 0

        for i, item in enumerate(training_instances):
            word_seq, word_seq_lengths = item[0].to(device), item[1].to(device)
            char_seq,char_seq_lengths,char_seq_recover = item[2].to(device),item[3].to(device),item[4].to(device)
            true_tags = item[5].to(device)
            mask = item[6].to(device)

            loss, pred_tag_seq = model.calculateLoss(data, word_seq, word_seq_lengths, \
                char_seq, char_seq_lengths,char_seq_recover, \
                true_tags, mask)

            mask = mask.cpu().data.numpy()
            pred = pred_tag_seq.cpu().data.numpy()
            gold = true_tags.cpu().data.numpy()
            overlaped = (pred == gold)
            right = np.sum(overlaped * mask)
            whole = mask.sum()
            right_token += right
            whole_token += whole
            sub_loss = loss.item()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()


            instances_nb = i*data.batch_size
            if instances_nb % 5000 == 0:
                print("Instance: {:05d}, sub loss: {:0.4f}; sub acc: {}/{} = {:0.4f}".format(instances_nb, sub_loss, right_token, whole_token,(right_token+0.)/whole_token))



        if epoch % 1 == 0:
            print("Epoch: {:04d}, total loss: {:0.4f}; total acc: {}/{} = {:0.4f}".format(epoch+1, total_loss, right_token, whole_token,(right_token+0.)/whole_token))

    torch.save(model, data.save_path)
    print("Training finished...")







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

    train(training_instances,data)



