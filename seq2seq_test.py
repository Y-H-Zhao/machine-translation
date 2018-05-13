# -*- coding: utf-8 -*-
"""
Created on Fri May 11 22:32:00 2018

@author: ZYH
"""

import tensorflow as tf
import codecs
import sys

#1.参数设置

# 读取checkpoint的路径。9000表示是训练程序在第9000步保存的checkpoint。
checkpoint_path = "./save/seq2seq_ckpt-9000"

# 模型参数。必须与训练时的模型参数保持一致。
hidden_size = 1024                         	# LSTM的隐藏层规模。
num_layers = 2                             	# 深层循环神经网络中LSTM结构的层数。
src_vocab_size = 10000                   	# 源语言词汇表大小。
trg_vocab_size = 4000                    	# 目标语言词汇表大小。
share_emb_and_softmax = True            	# 在Softmax层和词向量层之间共享参数。

# 词汇表文件
Src_vocab = "./en.vocab"
Trg_vocab = "./zh.vocab"

# 词汇表中<sos>和<eos>的ID。在解码过程中需要用<sos>作为第一步的输入，并将检查
# 是否是<eos>，因此需要知道这两个符号的ID。
sos_id = 1
eos_id = 2


#2.定义NMT模型和解码步骤
# 定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
           for _ in range(num_layers)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(hidden_size) 
           for _ in range(num_layers)])

        # 为源语言和目标语言分别定义词向量。   
        self.src_embedding = tf.get_variable(
            "src_emb", [src_vocab_size, hidden_size])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [trg_vocab_size, hidden_size])

        # 定义softmax层的变量
        if share_emb_and_softmax:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
               "weight", [hidden_size, trg_vocab_size])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [trg_vocab_size])

    def inference(self, src_input):
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里
        # 将输入句子整理为大小为1的batch。
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器。这一步与训练时相同。
        with tf.variable_scope("encoder",reuse = tf.AUTO_REUSE):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32)
   
        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        MAX_DEC_LEN=100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell",reuse = tf.AUTO_REUSE):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, sos_id)
            # 构建初始的循环状态。循环状态包含循环神经网络的隐藏状态，保存生成句子的
            # TensorArray，以及记录解码步数的一个整数step。
            init_loop_var = (enc_state, init_array, 0)

            # tf.while_loop的循环条件：
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), eos_id),
                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # 这里不使用dynamic_rnn，而是直接调用dec_cell向前计算一步。
                dec_outputs, next_state = self.dec_cell.call(
                    state=state, inputs=trg_emb)
                # 计算每个可能的输出单词对应的logit，并选取logit值最大的单词作为
                # 这一步的而输出。
                output = tf.reshape(dec_outputs, [-1, hidden_size])
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中。
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # 执行tf.while_loop，返回最终状态。
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()
        
#3.翻译一个测试句子
def main():
    # 定义训练用的循环神经网络模型。
    # 
    with tf.variable_scope("nmt_model", reuse=tf.AUTO_REUSE):
        model = NMTModel()

    # 定义个测试句子。
    test_en_text = input("请输入待翻译英文：")+"<eos>"
    #"This is a test ." "What are you doing ?" "I love you ." "I love their kids ."
    #"Who are you ?"
    '''
    原文的一些句子
    They 're actually looking at them down in that world .
    We 're going to take a joystick , sit in front of our computer , on the Earth , and press the joystick forward , and fly around the planet .
    We 're going to look at the mid-ocean ridge , a 40,000-mile long mountain range .
    The average depth at the top of it is about a mile and a half .
    And we 're over the Atlantic -- that 's the ridge right there -- but we 're going to go across the Caribbean , Central America , and end up against the Pacific , nine degrees north .
    We make maps of these mountain ranges with sound , with sonar , and this is one of those mountain ranges .
    We 're coming around a cliff here on the right .
    The height of these mountains on either side of this valley is greater than the Alps in most cases .
    And there 's tens of thousands of those mountains out there that haven 't been mapped yet .
    This is a volcanic ridge .
    We 're getting down further and further in scale .
    And eventually , we can come up with something like this .
    This is an icon of our robot , Jason , it 's called .
    And you can sit in a room like this , with a joystick and a headset , and drive a robot like that around the bottom of the ocean in real time .
    One of the things we 're trying to do at Woods Hole with our partners is to bring this virtual world -- this world , this unexplored region -- back to the laboratory .
    Because we see it in bits and pieces right now .
    '''
    print(test_en_text)
    
    # 根据英文词汇表，将测试句子转为单词ID。
    with codecs.open(Src_vocab, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in test_en_text.split()]
    print(test_en_ids)

    # 建立解码所需的计算图。
    output_op = model.inference(test_en_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    # 读取翻译结果。
    output_ids = sess.run(output_op)
    print(output_ids)
    
    # 根据中文词汇表，将翻译结果转换为中文文字。
    with codecs.open(Trg_vocab, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in output_ids])
    
    # 输出翻译结果。
    print(output_text.encode('utf8').decode(sys.stdout.encoding))
    sess.close()

if __name__ == "__main__":
    main()
