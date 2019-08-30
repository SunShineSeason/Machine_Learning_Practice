import collections
from operator import itemgetter
from parameters import *
import tensorflow as tf
from PTB_model  import *
import numpy as np
from data_process import *

# 使用给定的模型 model 在数据 batches 上运行 train_op 并返回在全部数据上的 perplexity 值:
def run_epoch(session, model,  dataset, train_op, step):
    # 计算平均perplexity的辅助变量。
    total_costs = 0.0
    iters = 0
    # 每个epoch 初始化第一个batch 时使用
    state = session.run(model.initial_state)

    iterator = dataset.make_initializable_iterator()
    batch = iterator.get_next()
    session.run(iterator.initializer)
    # 训练一个epoch。
    while True:
        try:
            x , y = session.run(batch)

            # 在当前batch上运行 train_op 并计算损失值。交叉熵损失函数计算的就是下一个单词为给定单词的概率。
            cost, state, _ = session.run(
                [model.cost, model.final_state, train_op],
                feed_dict={model.input_data: x, model.targets: y,
                           model.initial_state: state})
            # 注意，每一个 batch 输入模型时 lstm 单元的 初始状态都是上一个 batch 输出的终态 ！！！！！！！！
            total_costs += cost
            iters += model.num_steps

            # 只有在训练时输出日志。
            if step % 100 == 0:
                print("After %d steps, perplexity is %.3f" % (
                    step, np.exp(total_costs / iters)))
            step += 1
        except tf.errors.OutOfRangeError:
            break
    '''
    返回给定模型在给定数据上的 perplexity 值。 其中，total_costs / iters 表示整个数据集上每个样本在
    每个 lstm 神经单元上产生的损失的均值 ！！！！！！
    '''
    return step, np.exp(total_costs / iters)

# 使用数据集 id_list 产生 batches:
def make_batches(id_list, batch_size, num_step):
    dataset = tf.data.Dataset.from_tensor_slices(id_list)
    dataset = dataset.batch(num_step+1,drop_remainder=True)

    def split_sequence(sequence):
        inputs = sequence[:-1]
        labels = sequence[1:]
        return inputs,labels
    '''
    注意，搭建循环神经网络的时候 dataset.batch 参数 drop_remainder 要设置成 True ，因为循环神经单元状态的batch_size 是确定的 ！！
    '''
    dataset = dataset.map(split_sequence).batch(batch_size, drop_remainder=True)

    return dataset

''' 注意，本例中并没有运用这个语言模型产生 句子，测试集和验证集 是为了 训练和评估模型 ！！！！ '''
def main():
    global processed_data
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("language_model",
                           reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义测试用的循环神经网络模型。它与train_model共用参数，但是没有dropout。
    # 重点，模型是怎么进行 复用 的 ！！！！！！！！！！！！！！！！！
    with tf.variable_scope("language_model",
                            reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

            # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        step = 0
        dataset = make_batches(  # 注意，每次遍历训练集前要 重新初始化迭代器 ，因此这里最好使用 initializable iterator ！！！！！！！
            processed_data, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            step, train_pplx = run_epoch(session, train_model, dataset,
                                            train_model.train_op, step)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))


if __name__ == "__main__":
    main()