import codecs
import collections
from operator import itemgetter

''' 预处理的过程与 practice_projects/skip_gram/data_process.py 类似，值得对比学习 ！！ '''

VOCAB_SIZE=5000               # 词汇表的大小
RAW_DATA = "../dataset/ptb_data/data/ptb.train.txt"  # 训练集数据文件
VOCAB_OUTPUT = "ptb.vocab"                         # 输出的词汇表文件

with open(RAW_DATA,'r') as file:
    text=file.read().split()

counter=collections.Counter(text)
# 按词频顺序对单词进行排序:  (  一行代码实现，牛逼，值得学习 ！！！！ ！！！ )
# 使用 sorted(iterable[, cmp[, key[, reverse]]]) 函数，其中，
# operator.itemgetter(*items) :
# Return a callable object that fetches item from its operand using the operand’s __getitem__() method.
sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True)
vocab = [x[0] for x in sorted_word_to_cnt]
# print(type(itemgetter(1)))           -->>       <class 'operator.itemgetter'>

# 词典中加上一个换行符：
vocab = ["<eos>"] + vocab
# 删除低频词：( 更加高效的方法是在 第14 行 创建 频率字典的时候就直接过滤掉 低频词汇 )
vocab = vocab[:VOCAB_SIZE]

# 根据词典得到查找表 ( 一行代码实现 )：
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

# 如果出现了不在词汇表内的低频词，则替换为"unk"。
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

processed_data=[get_id(word) for word in text]
del text