import collections
from operator import itemgetter
from parameters import *
import tensorflow as tf

'''***************************************** 任务描述： **************************************************
使用 LSTM 结构为循环体结构( 未使用 Dropout ) 搭建循环神经网络 ( 用 TensorFlow 实现 )，根据 PTB 数据集生成语言模型
'''

# 通过一个 PTBModel 类来描述模型，这样方便维护循环神经网络中的状态。
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度( 即 时间步的大小 )。
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义每一步的输入和预期输出。两者的维度都是[batch_size, num_steps]。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        ''' 第一层：定义单词的词向量矩阵 '''
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转化为词向量,shape = ( batch_size, num_steps, HIDDEN_SIZE )
        #  注意， self.input_data 可以有 1 至多个维度 ！！！！！
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        ''' 第二层：定义使用 LSTM 结构为 循环体结构 的循环神经网络
         注意，本例中 lstm 单元输出向量的 维度 与 输入向量的 维度相等，都是 HIDDEN_SIZE !!!'''
        lstm = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
        # 初始化最初的状态，即全零的向量。这个量只在每个epoch 初始化第一个batch 时使用。
        self.initial_state = lstm.zero_state(batch_size, tf.float32)
        # print(type(self.initial_state))    --- >>>  class LSTMStateTuple

        # 定义输出列表。在这里先将不同时刻 LSTM 结构的输出收集起来，再一起提供给 softmax 层。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            '''Note: In order to make the learning process tractable, it is common practice to create an "unrolled" version of 
            the network, which contains a fixed number (num_steps) of LSTM inputs and outputs'''
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # cell_output.shape = ( batch_size, HIDDEN_SIZE )
                cell_output, state = lstm(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 把输出队列展开成[batch, hidden_size * num_steps] 的形状，然后再 reshape 成 [batch * numsteps, hidden_size] 的形状。
        # 注意，这里要保证 output 每一行表示的样本 与 self.targets 中样本的顺序要一致 ！！
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        ''' 第三层：定义 softmax 层： '''
        # Softmax层：将 RNN 在每个位置上的输出转化为各个单词的logits。
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        # tf.matmul(output, weight).shape = ( batch * numsteps , vocab_size )
        logits = tf.matmul(output, weight) + bias

        ''' 定义损失函数：注意，self.cost 计算的 公式,没有 直接 reduce_mean() ！！！！！
        cost 表示 一个 batch 中每个样本在 num_steps 个时间步上计算的总的损失 的均值
         '''
        # loss.shape = ( batch_size * numsteps )
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not is_training:
            return

        ''' 执行梯度截断 并定义 优化算法（ 使用 梯度下降 ）：
        -----------------------------------------------------------------------------------------------------
        tf.gradients(ys,xs) 实现 ys 对 xs 求导 ，求导返回值是一个list，list 的长度等于len(xs)
        假设返回值是[grad1, grad2, grad3]，ys=[y1, y2]，xs=[x1, x2, x3]。则：grad1 为 y1 和 y2 对 x1 求导的张量的和
        -------------------------------------------------------------------------------------------------------
        tf.clip_by_global_norm(t_list,clip_norm) 对 t_list 列表中的每一个张量 分别执行梯度截断，返回 截断后的 梯度列表（list_clipped） 和 全局范数 （global_norm）           
        '''
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(         # 执行梯度截断，参考 Pascanu et al., 2012 (pdf)论文
            tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))
        print(trainable_variables)
        print('Note,the gradients of the trainable_variables are of the same shape as them:\n',grads)
        print('*'*50,'The model has been built','*'*50)
PTBModel(True,12,8)

