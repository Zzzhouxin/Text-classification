"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 15:45
# @Author  : zhouxin
# @FileName: load_model.py
# @Software: PyCharm
===========================
"""
import os
import pickle
import pickle
import torch
from utils import build_vocab, MAX_VOCAB_SIZE
from utils_fasttext import build_dataset, build_iterator, get_time_dif, DatasetIterater
from importlib import import_module

device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')


with open('./dscan_result/saved_dict/models.pkl', 'rb') as f:
    model = pickle.load(f)

test_data ="HTTP/1.0 200 OK Vary: Accept-Encoding Content-Type: text/html Accept-Ranges: bytes ETag: ETAG Last-Modified: LAST-MODIFIED Expires: Fri  22 Dec 2023 18:45:00 GMT Cache-Control: max-age=31104000 Content-Length: CONTENT-LENGTH Connection: close Data: DATE Server: lighttpd/1.4.45 Swegon GOLD HTTP/1.0 416 Requested Range Not Satisfiable Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close Data: DATE Server: lighttpd/1.4.45 416 - Requested Range Not Satisfiable 416 - Requested Range Not Satisfiable   HTTP/1.0 404 Not Found Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close Data: DATE Server: lighttpd/1.4.45  HTTP/1.0 411 Length Required Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close Data: DATE Server: lighttpd/1.4.45 411 - Length Required 411 - Length Required   HTTP/1.0 403 Forbidden Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close Data: DATE Server: lighttpd/1.4.45 403 - Forbidden 403 - Forbidden"

model.to(device)
model.eval()


# 加载或构建词汇表
def load_vocab(vocab_path):
    if os.path.exists(vocab_path):
        vocab = pickle.load(open(vocab_path, 'rb'))
    else:
        # 这里需要有构建词汇表的数据文件路径和分词方法
        vocab = build_vocab('path_to_train_data', lambda x: x.split(), MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(vocab, open(vocab_path, 'wb'))
    return vocab

# 处理单条数据
def process_data(data, vocab, pad_size=512):
    UNK, PAD = '<UNK>', '<PAD>'  # 未知字和padding符号，从utils.py中获取
    words = data.split()  # 分词，根据实际情况调整
    words_line = []
    seq_len = len(words)

    # 截断或填充
    if len(words) < pad_size:
        words.extend([PAD] * (pad_size - len(words)))
    else:
        words = words[:pad_size]
        seq_len = pad_size

    # 转换为词索引
    for word in words:
        words_line.append(vocab.get(word, vocab.get(UNK)))

    # 转换为tensor，注意这里仅处理单条数据，故不使用batch处理
    tensor_data = torch.LongTensor([words_line])
    return (tensor_data, torch.LongTensor([seq_len]))

# 使用示例
vocab_path = './dscan_result/data/vocab.pkl'  # 词汇表路径
vocab = load_vocab(vocab_path)  # 加载词汇表
new_data = "HTTP/1.1 400 Bad Request Data: DATE Server: Apache Connection: close Content-Type: text/html; charset=utf-8 DataComm Computers & Networks Server DataComm Computers & Networks Server Bad Request HTTP/1.1 400 Bad Request Data: DATE Server: Apache Connection: close Content-Type: text/html; charset=utf-8 DataComm Computers & Networks Server DataComm Computers & Networks Server Bad Request HTTP/1.1 400 Bad Request Data: DATE Server: Apache Connection: close Content-Type: text/html; charset=utf-8 DataComm Computers & Networks Server DataComm Computers & Networks Server Bad Request HTTP/1.1 400 Bad Request Data: DATE Server: Apache Connection: close Content-Type: text/html; charset=utf-8 DataComm Computers & Networks Server DataComm Computers & Networks Server Bad Request HTTP/1.1 400 Bad Request Data: DATE Server: Apache Connection: close Content-Type: text/html; charset=utf-8 DataComm Computers & Networks Server DataComm Computers & Networks Server Bad Request HTTP/1.1 400 Bad Request Data: DATE Server: Apache Connection: close Content-Type: text/html; charset=utf-8 DataComm Computers & Networks Server DataComm Computers & Networks Server Bad Request HTTP/1.1 400 Bad Request Data: DATE Server: Apache Connection: close Content-Type: text/html; charset=utf-8 DataComm Computers & Networks Server DataComm Computers & Networks Server Bad Request"
processed_data = process_data(new_data, vocab)

# 之后可以将processed_data传递到模型中进行预测

tensor_data, tensor_seq_len = process_data(new_data, vocab)
tensor_data = tensor_data.to(device)
tensor_seq_len = tensor_seq_len.to(device)


# 模型推断
with torch.no_grad():
    outputs = model((tensor_data, tensor_seq_len))
    prediction = torch.max(outputs, 1)[1].cpu().numpy()


print(prediction)


class_list = [x.strip() for x in open('./dscan_result/data/class.txt', encoding='utf-8').readlines()]

print(class_list[prediction[0]])

