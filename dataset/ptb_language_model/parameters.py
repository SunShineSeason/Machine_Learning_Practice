TRAIN_DATA = "ptb.train"          # 训练数据路径。
EVAL_DATA = "ptb.valid"           # 验证数据路径。
TEST_DATA = "ptb.test"            # 测试数据路径。
HIDDEN_SIZE = 300                 # 隐藏层规模。
NUM_LAYERS = 2                    # 深层循环神经网络中LSTM结构的层数。
VOCAB_SIZE = 10000                # 词典规模。
TRAIN_BATCH_SIZE = 20             # 训练数据batch的大小。
TRAIN_NUM_STEP = 35               # 训练数据截断长度。

EVAL_BATCH_SIZE = 1               # 测试数据batch的大小。
EVAL_NUM_STEP = 1                 # 测试数据截断长度。
NUM_EPOCH = 5                     # 使用训练数据的轮数。
LSTM_KEEP_PROB = 0.9              # LSTM节点不被dropout的概率。
EMBEDDING_KEEP_PROB = 0.9         # 词向量不被dropout的概率。
MAX_GRAD_NORM = 5                 # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True      # 在Softmax层和词向量层之间共享参数。