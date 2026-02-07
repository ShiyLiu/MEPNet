import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""
    def __init__(self, args):
        self.model_name = 'MEPNet'
        self.train_path = args.dataset + '/data/train.csv'                                  # 训练集
        self.dev_path = args.dataset + '/data/dev.csv'                                      # 验证集
        self.test_path = args.dataset + '/data/test.csv'                                    # 测试集
        self.class_list = [x.strip() for x in open(
            args.dataset + '/data/class.txt', encoding='utf-8').readlines()]                # 类别名单
        self.vocab_path = args.dataset + '/data/vocab.pkl'                                  # 词表
        self.save_path = args.dataset + '/saved_dict/' + self.model_name + '.ckpt'          # 模型训练结果
        self.log_path = args.dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(args.dataset + '/data/' + args.embedding)["embeddings"].astype('float32'))\
            if args.embedding != 'random' else None                                         # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')          # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 256                                           # mini-batch大小
        self.pad_size = 128                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 64           # 字向量维度

        self.filter_sizes = (6, 7, 8)                                   # 卷积核尺寸
        self.num_filters1 = 256                                         # 第一层嵌入卷积核数量(channels数)
        self.num_filters2 = 256                                        # 第二层嵌入卷积核数量(channels数)
        self.bit_dim = 32                                               # 位嵌入的维度

        self.middle = 20
        self.d_model = self.middle * 2
        self.num_heads = 10


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, q,k,v):
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformation
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # Split into heads
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores += mask.unsqueeze(1).unsqueeze(1)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)

        # Concatenate heads and linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)   # [batch_size, len(filter_size), num_filters]

        return output


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters1, (k, config.embed)) for k in config.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters2, (k, config.bit_dim)) for k in config.filter_sizes])

        self.linear = nn.Linear(config.num_filters1 * len(config.filter_sizes) * 2,  config.middle * 2)
        # self.linear1 = nn.Linear(config.num_filters1 * len(config.filter_sizes),  config.middle)
        # self.linear2 = nn.Linear(config.num_filters2 * len(config.filter_sizes),  config.middle)

        self.attention = MultiHeadAttention(config.d_model, config.num_heads)
        # self.attention = SelfAttention(config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.middle * 2, config.num_classes)

    def forward(self, x):
        out1 = self.embedding(x[0])    # [batch_size, seq_len, embedding_dim]
        out1 = out1.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        # Convolutional layers
        conv_out1 = [F.relu(conv(out1)).squeeze(3) for conv in self.convs1]  # [batch_size, num_filters, seq_len-fliter_size+1]
        pooled_out1 = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in conv_out1]   # [batch_size,num_filters]
        # out1 = torch.cat([_t.unsqueeze(1) for _t in pooled_out1], 1)   # [batch_size, len(filter_size), num_filters]
        out1 = torch.cat([_t for _t in pooled_out1], 1)  # [batch_size, len(filter_size) * num_filters]
        # out1 = F.relu(self.linear1(out1))

        out2 = x[1]  # [batch_size, seq_len, embedding_dim]
        out2 = out2.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        conv_out2 = [F.relu(conv(out2)).squeeze(3) for conv in self.convs2]  # [batch_size, num_filters, seq_len-fliter_size+1]
        pooled_out2 = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in
                       conv_out2]  # [batch_size,num_filters]
        # out2 = torch.cat([_t.unsqueeze(1) for _t in pooled_out2], 1)  # [batch_size, len(filter_size), num_filters]
        out2 = torch.cat([_t for _t in pooled_out2], 1)  # [batch_size, len(filter_size) * num_filters]
        # out2 = F.relu(self.linear2(out2))

        out3 = torch.cat([out2, out1], 1)
        out3 = F.relu(self.linear(out3))

        # Attention layer
        out3 = out3.unsqueeze(1)    # [batch_size, seq_len, d_model] [batch_size, len(filter_size), num_filters]
        out3 = self.attention(out3, out3, out3)
        out3 = out3.view(out3.shape[0], -1)

        # Fully connected layer
        out3 = self.dropout(out3)
        out3 = self.fc(out3)
        out3 = torch.sigmoid(out3)

        return out3