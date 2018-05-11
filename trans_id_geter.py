# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:42:36 2018

@author: ZYH
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:00:33 2018

@author: ZYH
"""
import codecs #codecs专门用作编码转换 防止乱码
#import sys 
#最后进行训练的是那些转化为单词编号数据

MODE = "TRANSLATE_ZH"    # 将MODE设置为"TRANSLATE_EN", "TRANSLATE_ZH"之一。


if MODE == "TRANSLATE_ZH":   # 中文翻译数据
    raw_data = "train.txt.zh"
    vocab_data = "zh.vocab"
    output_data = "train.zh"
elif MODE == "TRANSLATE_EN":   # 英文翻译数据
    raw_data = "train.txt.en"
    vocab_data = "en.vocab"
    output_data = "train.en"


'''
raw_data="ptb.train.txt" #原始数据
vocab_data="ptb.vocab" #词汇表文件
output_data="ptb.train" #将单词替换为单词编号后的输出文件
'''
#读取词汇表，并建立词汇到编号的映射
with codecs.open(vocab_data,"r","utf-8") as f_vocab:
    vocab=[w.strip() for w in f_vocab.readlines()]
word_to_id={k:v for (k,v) in zip(vocab,range(len(vocab)))}

#如果出现了未知 或者被删除的词汇 替换为“<unk>”
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]
f_in=codecs.open(raw_data,"r","utf-8")
f_out=codecs.open(output_data,"w","utf-8")

for line in f_in:
    words=line.strip().split()+["<eos>"] #读取单词并添加结束符
    #将单词替换成词汇表中的编号
    out_line=' '.join(str(get_id(w)) for w in words) +"\n"
    f_out.write(out_line)
    
f_in.close()
f_out.close()

#以上示例以文本文件保存处理过的数据，实际工程中，通常使用TFRecords格式来提高读写效率
