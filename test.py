import torch
from transformers import BertModel, BertTokenizer

pretrained_path = './bert-base-uncased'
# 从文件夹中加载bert模型
model = BertModel.from_pretrained(pretrained_path)
# 从bert目录中加载词典
tokenizer = BertTokenizer.from_pretrained(pretrained_path)

# 输出字典的大小
print(f'vocab size :{tokenizer.vocab_size}')
# 把'[PAD]'编码
print(tokenizer.encode('[PAD]'))
# 把'[SEP]'编码
print(tokenizer.encode('[SEP]'))
