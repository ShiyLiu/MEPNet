# coding: UTF-8
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
import pickle as pkl
import os


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    cur_batch = 0                   # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0                # 记录上次验证集loss下降的batch数
    flag = False                    # 记录是否很久没有效果提升
    log_dir = config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())
    writer = SummaryWriter(log_dir=log_dir)

    train_loss_list = []
    train_acc_list = []

    dev_loss_list = []
    dev_acc_list = []

    t1 = time.time()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        train_loss = 0.0
        train_correct = 0
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            model.train()
            # trains = trains.clone().detach()
            optimizer.zero_grad()
            outputs = model(trains)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

            cur_batch += 1

        dev_acc, dev_loss = evaluate(config, model, dev_iter)

        train_loss = train_loss / len(train_iter.batches)
        train_acc = train_correct / len(train_iter.batches)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        dev_loss_list.append(dev_loss)
        dev_acc_list.append(dev_acc)

        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = epoch + 1
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
        print(msg.format(cur_batch, train_loss, train_acc, dev_loss, dev_acc, time_dif, improve))

        # if (epoch + 1) - last_improve > 5:
        #     # 验证集loss超过一段时间没下降，结束训练
        #     print("No optimization for a long time, auto-stopping...")
        #     # flag = True
        #     break

        # if flag:
        #     break
    writer.close()

    t2 = time.time()
    print("Time usage for training: ", '%.3f s, ' % (t2 - t1), timedelta(seconds=int(round(t2 - t1))))

    test_acc, test_loss, precision, recall, f1_score, confusion = test(config, model, test_iter)
    info = {}
    info.update({'model': config.model_name})
    info.update({'train_acc': train_acc_list})
    info.update({'train_loss': train_loss_list})
    info.update({'dev_acc': dev_acc_list})
    info.update({'dev_loss': dev_loss_list})
    info.update({'test_acc': test_acc})
    info.update({'test_loss': test_loss})
    info.update({'precision': precision})
    info.update({'recall': recall})
    info.update({'f1_score': f1_score})
    info.update({'confusion': confusion})
    info.update({'best_epoch': last_improve})
    pkl.dump(info, open(log_dir+'/info.pkl', 'wb'))


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc, test_loss, precision, recall, f1_score, confusion = evaluate(config, model, test_iter, test=True)
    return test_acc, test_loss, precision, recall, f1_score, confusion


def evaluate(config, model, data_iter, test=False):
    t1 = time.time()
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    val_loss = 0.0
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)

            val_loss += loss.item()

            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # time_dif = get_time_dif(t1)
    t2 = time.time()

    val_loss /= len(data_iter.batches)
    val_acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        accuracy = metrics.accuracy_score(y_true=labels_all, y_pred=predict_all)
        precision = metrics.precision_score(y_true=labels_all, y_pred=predict_all, average='binary')
        recall = metrics.recall_score(y_true=labels_all, y_pred=predict_all, average='binary')
        f1_score = metrics.f1_score(y_true=labels_all, y_pred=predict_all, average='binary')
        confusion = metrics.confusion_matrix(labels_all, predict_all)

        print('accuracy', "%.4f" % accuracy)
        print('precision', "%.4f" % precision)
        print('recall', "%.4f" % recall)
        print('f1', "%.4f" % f1_score)
        print("Confusion Matrix...")
        print(confusion)
        print("Precision, Recall and F1-Score...")
        print(report)

        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(val_loss, accuracy))
        # print("Time usage:", time_dif)
        print("Time usage for testing: ", '%.3f s, ' % (t2 - t1), timedelta(seconds=int(round(t2 - t1))))

        return val_acc, val_loss, precision, recall, f1_score, confusion
    return val_acc, val_loss


if __name__ == '__main__':
    vocab = pkl.load(open('info.pkl', 'rb'))
    print()

