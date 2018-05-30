import MeCab
import sys
import pandas as pd
import pdb
import numpy as np
from gensim.models import word2vec
import logging
import os
import xml.etree.ElementTree as ET
import pickle as pkl
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class livedoor_w2v:
    def __init__(self,file_name,id):
        self.fn_in = [self.make_path(['./livedoor-news-data','{}.xml'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_in_txt = [self.make_path(['./livedoor-news-data-txt','{}.txt'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_out = [self.make_path(['./livedoor-news-data-wakati','{}.xml'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_model = self.make_path(['./livedoor-news-data-model','word2vec.model'])
        self.fn_pkl = [self.make_path(['./livedoor-news-pkl','{}.pickle'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.id = id

    def wakati(self,file_name_in,file_name_out):#分かち書きに変換して保存する
        tagger = MeCab.Tagger('-Ochasen')
        fi = open(file_name_in,'r')
        fo = open(file_name_out,'w')

        line = fi.readline()
        while line:
            word = tagger.parse(line).split('\n')
            result = [x.split('\t')[0] for x in word]
            for x in result:
                if '　' in x:
                    x = x.replace('　','_')
                if ' ' in x:
                    x = x.replace(' ','_')
                fo.write(x + ' ')
            fo.write('\n')
            line = fi.readline()

        fi.close()
        fo.close()

    def w2v_train(self):#word2vecを学習する
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)
        sentences = word2vec.PathLineSentences('./livedoor-news-data-wakati')
        model = word2vec.Word2Vec(sentences,sg = 1,size = 100,min_count = 1,window = 10,hs = 1,negative = 0)
        model.save(self.fn_model)

    def mean_w2v_dic(self):#すべての単語についてベクトルを出力し、平均を求める
        model = word2vec.Word2Vec.load(self.fn_model)
        word_vec = np.array([model.wv[self.dic[0]]])
        for i in np.arange(1,len(self.dic)):
            try:
                word_vec = np.vstack([word_vec,model.wv[self.dic[i]].reshape(1,100)])
            except:
                print('word_vec_ERROR! errorWord : {}'.format(self.dic[i]))
                continue

        self.word_vec_ave = np.mean(word_vec,axis=0)
        
        #self.word_vec = np.mean(np.array([model.wv[word] for word in self.dic]),axis=0)

    def mean_w2v(self,fname):#渡されたファイルの中にあるすべての単語についてベクトルを出力し、平均を求める
        number_count = 0
        model = word2vec.Word2Vec.load(self.fn_model)
        fp = open(fname,'r')
        word_strList = fp.read().split('\n')
        vec_aveList = np.array([])
        skip_strlist = ['',' ','　']
        for x in word_strList:
            if x in skip_strlist:
                continue
            word_vec = np.array([])
            word_list = x.split(' ')
            for word in word_list:
                try:
                    if word_vec.shape[0] == 0:
                        word_vec = model.wv[word]
                    else:
                        word_vec = np.vstack([word_vec,model.wv[word]])
                except:
                    print('word_vec_ERROR! errorWord : {}'.format(word))
                    continue

            if vec_aveList.shape[0] == 0:
                vec_aveList = np.mean(word_vec,axis = 0)
            else:
                vec_aveList = np.vstack([vec_aveList,np.mean(word_vec,axis = 0).reshape(1,100)])

        return vec_aveList

    def make_path(self,path):#パスの作成(引数pathリストに入っているものを接続する)
        if not os.path.exists(path[0]):
            os.makedirs(path[0])

        for i in np.arange(len(path)-1):
            path[i+1] = os.path.join(path[i],path[i+1])
        return path[len(path)-1]

    def xml_shaping(self,input,output):#xmlから記事内容のみを抽出したものを保存する
        tree = ET.parse(input)
        root = tree.getroot()
        text = []
        for i in np.arange(len(root)):
            str_array = np.array([root[i][ind].text for ind in np.arange(3,len(root[i]))])
            str_array = np.delete(str_array,np.where(str_array == None))
            str_df = pd.DataFrame(str_array)
            url_ind = np.where(str_df[0].str.contains('http').values)
            str_list = list(np.delete(str_array,url_ind))
            text.append(str_list)
        f = open(output,'w')

        for c in text:
            for bff in c:
                f.write(bff)
            f.write('\n')
        f.close()

    def make_dic(self,input_file,dic_file,flag):#辞書データ読み込み,追加,作成,更新
        mecab = MeCab.Tagger("-Ochasen")
        skip_strList = [' ','','　']
        if not os.path.isdir('data'):
            os.makedirs('data')
        #辞書データが作られて居なければ書き込みモードで起動してデータの入れ物を作成
        if not os.path.isfile(dic_file):
            f = open(dic_file,'w')
            self.dic = []
        else:#作られて居たならば中身を読み込む
            f = open(dic_file,'r')
            self.dic = f.read().split('\n')
        f.close()

        if flag:
            #xmlを整形して作ったtxtから単語を読み出す
            f_in = open(input_file,'r')
            word = []
            buf = f_in.read().split('\n')
            for x in buf:
                mor = mecab.parse(x).split('\n')
                for w in mor:
                    pdb.set_trace()
                    if word == []:
                        word = list(w.split('\t')[0])
                    else:
                        word.append(w.split('\t')[0])
            f_in.close()
            count = 0
            #辞書データの中にwordが含まれて居なかったら追加
            for x in word:
                if x in skip_strList:
                    continue

                if count % 10000 == 0:
                    print('word:{}/{}'.format(count,len(word)))

                if self.dic == []:
                    self.dic = list(x)
                else:
                    df = pd.DataFrame(self.dic)
                    if np.sum((df[0] == x).values) == 0:
                        self.dic.append(x)
                count+=1
            #辞書データを改行区切りで書き出す
            f_dict = open(dic_file,'w')
            for x in self.dic:
                f_dict.write(x + '\n')
            f_dict.close()

if __name__ =="__main__":
    Data_flag = (sys.argv[1]=='T')
    dict_flag = (sys.argv[2]=='T')
    train_flag = (sys.argv[3]=='T')
    #ファイル名をまとめているテキストからファイル名を読み出す
    f = open('file_name.txt','r')
    #ファイル名がすべて接続されて居るので区切り文字を指定してファイル名ごとに分ける
    lines = f.readlines()
    fn = [line.strip() for line in lines]
    dic_path = './data/dict.txt'
    #オブジェクト作成
    w2v = livedoor_w2v(fn,len(fn))
    if Data_flag:
        for i in np.arange(w2v.id):
            print("inputfile:{}".format(w2v.fn_in[i]))
            print("outputfile:{}".format(w2v.fn_out[i]))
            w2v.xml_shaping(w2v.fn_in[i],w2v.fn_in_txt[i])
            w2v.wakati(w2v.fn_in_txt[i],w2v.fn_out[i])
            if dict_flag:
                w2v.make_dic(w2v.fn_in_txt[i],dic_path,Data_flag)
    else:
        w2v.make_dic(w2v.fn_in_txt[0],dic_path,False)

    if train_flag:
        w2v.w2v_train()

    for i in np.arange(w2v.id):
        vec_ave = w2v.mean_w2v(w2v.fn_out[i])
        fp = open(w2v.fn_pkl[i],'wb')
        pkl.dump(vec_ave,fp)
        fp.close()
    #pk.dump(w2v.word_vec_ave,fp)
    #fp.close()
