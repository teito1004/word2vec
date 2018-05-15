import MeCab
import sys
import pandas as pd
import pdb
import numpy as np
from gensim.models import word2vec
import logging
import os

class livedoor_w2v:
    def __init__(self,file_name,id):
        self.fn_in = [self.make_path(['./livedoor-news-data','{}.xml'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_out = [self.make_path(['./livedoor-news-data-wakati','{}.xml'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_model = self.make_path(['./livedoor-news-data-model','word2vec.model'])
        self.id = id

    def wakati(self,file_name_in,file_name_out):#分かち書きに変換して保存する
        tagger = MeCab.Tagger('-F\s%f[6] -U\s%m -E\\n')
        fi = open(file_name_in,'r')
        fo = open(file_name_out,'w')

        line = fi.readline()
        while line:
            result = tagger.parse(line)
            fo.write(result[1:])
            line = fi.readline()

        fi.close()
        fo.close()

    def w2v_train(self):#word2vecを学習する
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)
        sentences = word2vec.PathLineSentences('./livedoor-news-data-wakati')
        model = word2vec.Word2Vec(sentences,sg = 1,size = 100,min_count = 1,window = 10,hs = 1,negative = 0)
        model.save(self.fn_model)

    #def mean_w2v(self):#すべての単語についてベクトルを出力し、平均を求める

    def make_path(self,path):#パスの作成(引数pathリストに入っているものを接続する)
        if not os.path.exists(path[0]):
            os.makedirs(path[0])

        for i in np.arange(len(path)-1):
            path[i+1] = os.path.join(path[i],path[i+1])
        return path[len(path)-1]

if __name__ =="__main__":
    #ファイル名をまとめているテキストからファイル名を読み出す
    f = open('file_name.txt','r')
    #ファイル名がすべて接続されて居るので区切り文字を指定してファイル名ごとに分ける
    lines = f.readlines()
    fn = [line.strip() for line in lines]
    print(fn)
    #オブジェクト作成
    w2v = livedoor_w2v(fn,len(fn))

    print(w2v.id)
    for i in np.arange(w2v.id):
        print(i)
        print(w2v.fn_in[i])
        print(w2v.fn_out[i])
        w2v.wakati(w2v.fn_in[i],w2v.fn_out[i])

    w2v.w2v_train()

