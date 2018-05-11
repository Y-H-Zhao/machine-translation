# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:08:46 2018

@author: ZYH
"""

import codecs #codecs专门用作编码转换 防止乱码
import collections 
from operator import itemgetter

MODE = "TRANSLATE_EN"    # 将MODE设置为"TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    raw_data = "train.txt.zh"
    vocab_output = "zh.vocab"
    vocab_size = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    raw_data = "train.txt.en"
    vocab_output = "en.vocab"
    vocab_size = 10000


counter = collections.Counter() #统计单词频率
with codecs.open(raw_data,"r","utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word]+=1
#按词频对单词排序
sort_word_to_cnt=sorted(counter.items(),
                        key=itemgetter(1),
                        reverse=True)
sorted_words=[x[0] for x in sort_word_to_cnt]
'''
#稍后我们需要在文本换行处加入
“<eos>”句子结束符,“<sos>”句子起始符，“<unk>”低频替代符,
这里预先将其加入词汇表
'''
sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
if len(sorted_words) > vocab_size:
    sorted_words = sorted_words[:vocab_size]

with codecs.open(vocab_output,'w','utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word +"\n")
