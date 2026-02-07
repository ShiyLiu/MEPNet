# coding: UTF-8
import os
# os.environ['CUDNN_CONV_ALGO'] = '1'  # 1=FFT, 2=Winograd等（具体值需查cuDNN文档）
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["TORCH_LOGS"] = "+dynamo"
import random
import time
import numpy as np
from datetime import timedelta
import torch
from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Phishing URL Detection')
parser.add_argument('--model', default='MEPNet', type=str,
                    help='Choose a model: MEPNet')
parser.add_argument('--dataset', type=str, default='ISCXURL2016', help='Choose a dataset: Ebbu2017, PhishStorm, Martinez, ISCXURL2016')
parser.add_argument('--embedding_mode', default='random', type=str, help='random or pre_trained')
parser.add_argument('--embedding', default='embedding_SougouNews.npz', type=str, help='Paths for pre-trained word vectors')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')

args = parser.parse_args()


if __name__ == '__main__':

    if args.embedding_mode == 'random':
        args.embedding = 'random'

    model_name = args.model         # 模型名字
    dataset = args.dataset          # 数据集
    embedding = args.embedding      # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random

    if model_name in ['MEPNet']:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(args)                                 # 网络模型的对应参数
    seed = 8
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = True                   # 保持cuDNN启用
    torch.backends.cudnn.benchmark = True                   # 自动优化卷积运算
    torch.backends.cudnn.deterministic = True               # 保证每次结果一样
    print(f"benchmark: {torch.backends.cudnn.benchmark}")
    print(f"deterministic: {torch.backends.cudnn.deterministic}")

    print(f'The model being trained is {config.model_name}')
    print(f'The dataset being used is {args.dataset}')

    print("Loading data...")
    t1 = time.time()

    train_data, dev_data, test_data = build_dataset(config, args.word)  # 数据预处理
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    t2 = time.time()
    print("Time usage for data preprocessing: ", '%.3f s, ' % (t2-t1), timedelta(seconds=int(round(t2-t1))))

    # train
    model = x.Model(config).to(config.device)
    
    print(model.parameters)
    t3 = time.time()
    train(config, model, train_iter, dev_iter, test_iter)
    t4 = time.time()
    print("Time usage for training: ", '%.3f s, ' % (t4 - t3), timedelta(seconds=int(round(t4 - t3))))