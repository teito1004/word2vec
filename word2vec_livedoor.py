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

class makeData:
    def __init__(self,file_name,id):
        self.fn_in = [self.make_path(['./livedoor-news-data','{}.xml'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_in_txt = [self.make_path(['./livedoor-news-data-txt','{}.txt'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_out = [self.make_path(['./livedoor-news-data-wakati','{}.xml'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_model = self.make_path(['./livedoor-news-data-model','word2vec.model'])
        self.fn_w2v = [self.make_path(['./livedoor-news-data-pkl','w2v','{}.pickle'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.fn_tfidf = [self.make_path(['./livedoor-news-data-pkl','tfidf','{}.pickle'.format(file_name[i])]) for i in np.arange(len(file_name))]
        self.id = id

    def make_path(self,path):#パスの作成(引数pathリストに入っているものを接続する)
        for i in np.arange(len(path)-1):
            #path途中のフォルダがなければ作る
            if not os.path.exists(path[i]):
                os.makedirs(path[i])
            #pathを接続する
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

    def wakati(self,file_name_in,file_name_out):#分かち書きに変換して保存する
        tagger = MeCab.Tagger('-Ochasen')
        fi = open(file_name_in,'r')
        fo = open(file_name_out,'w')

        line = fi.readline()
        while line:
            word = tagger.parse(line).split('\n')
            result = [x.split('\t')[0] for x in word]
            for x in result:
                if ' ' in x:
                    x = x.replace(' ','_')

                fo.write(x + ' ')
            fo.write('\n')
            line = fi.readline()

        fi.close()
        fo.close()

    def make_dic(self,input_file,dic_file):#辞書データ読み込み,追加,作成,更新
        skip_strList = [' ','','　']
        if not os.path.isdir('data'):
            os.makedirs('data')
        #辞書データが作られて居なければ書き込みモードで起動してデータの入れ物を作成
        if not os.path.isfile(dic_file):
            dic = []
        else:#作られて居たならば中身を読み込む
            dic = list(pd.read_csv(dic_file)['word'].values)

        #idf値に必要なものの宣言
        sentence_cnt = 0
        word_sentence_cnt = np.zeros(len(dic))
        word_allcnt = np.zeros(len(dic))

        for fn in input_file:
            #xmlを整形して作ったtxtから単語を読み出す
            f_in = open(fn,'r')
            #記事ごとに分割したリストを作成
            buf = f_in.read().split('\n')
            for x in buf:
                sentence_cnt+=1
                #単語ごとに分割したリストを作成
                mor = x.split(' ')
                #記事内で重複要素を消したリストを作成
                mor_list = list(set(mor))
                for w in mor_list:
                    if w in skip_strList:
                        continue
                    if w in dic:
                        dic.append(w)
                        word_allcnt = np.append(word_allcnt,np.array([1]))
                        word_sentence_cnt = np.append(word_sentence_cnt,np.array([1]))
                    else:
                        word_allcnt[np.where(np.array(dic) == w)[0]]+= mor.count(w)
                        word_sentence_cnt[np.where(np.array(dic) == w)[0]]+=1
            f_in.close()
        idf = np.log(sentence_cnt)-np.log(word_sentence_cnt)+1
        df = pd.DataFrame(np.hstack([np.array(dic).reshape(-1,1),word_allcnt.reshape(-1,1),word_sentence_cnt.reshape(-1,1),idf.reshape(-1,1)]),columns = ['word','word_cnt_all','word_cnt_sentence','idf'])
        df.to_csv(dic_file)

class word2vec_livedoor:
    def __init__(self,datapath,modelpath,dictpath):
        self.datapath = datapath
        self.modelpath = modelpath
        self.dictTable = pd.read_csv(dictpath,index_col = 0)

    def w2v_train(self):#word2vecを学習する
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)
        sentences = word2vec.PathLineSentences('./livedoor-news-data-wakati')
        model = word2vec.Word2Vec(sentences,sg = 1,size = 100,min_count = 1,window = 10,hs = 1,negative = 0)
        model.save(self.modelpath)

    def mean_w2v(self,outpath):#渡されたファイルの中にあるすべての単語についてベクトルを出力し、平均を求める
        model = word2vec.Word2Vec.load(self.modelpath)
        for i in np.arange(len(self.datapath)):
            fp = open(self.datapath[i],'r')
            sentenceList = fp.read().split('\n')
            vec_aveList = np.array([])
            for sentence in sentenceList:
                word_strList = sentence.split(' ')
                word_vec = np.array([])
                for word in word_strList:
                    if word == '':
                        continue
                    try:
                        if word_vec.shape[0] == 0:
                            word_vec = model.wv[word]
                        else:
                            word_vec = np.vstack([word_vec,model.wv[word]])
                    except:
                        print('word_vec_ERROR! errorWord : {}'.format(word))
                        continue

                if word_vec.shape[0]==0:
                    continue

                if vec_aveList.shape[0] == 0:
                    vec_aveList = np.mean(word_vec,axis = 0)
                else:
                    vec_aveList = np.vstack([vec_aveList,np.mean(word_vec,axis = 0)])

            fout = open(outpath[i],'wb')
            pkl.dump(vec_aveList,fout)
            fout.close()

    def mean_w2v_tfidf(self,outpath):#渡されたファイルの中にあるすべての単語についてベクトルを出力したものにTFIDFを適応し、平均を求める
        model = word2vec.Word2Vec.load(self.modelpath)
        for i in np.arange(len(self.datapath)):
            fp = open(self.datapath[i],'r')
            sentenceList = fp.read().split('\n')
            vec_aveList = np.array([])
            for sentence in sentenceList:
                word_strList = sentence.split(' ')
                word_vec = np.array([])
                for word in word_strList:
                    try:
                        tf = word_strList.count(word)/len(word_strList)
                        idf = self.dictTable.at[word,'idf']
                        tf_idf = tf*idf
                        if word_vec.shape[0] == 0:
                            word_vec = model.wv[word]*tf_idf
                        else:
                            word_vec = np.vstack([word_vec,model.wv[word]*tf_idf])
                    except:
                        print('word_vec_ERROR! errorWord : {}'.format(word))
                        continue

                if word_vec.shape[0]==0:
                    continue

                if vec_aveList.shape[0] == 0:
                    vec_aveList = np.mean(word_vec,axis = 0)
                else:
                    vec_aveList = np.vstack([vec_aveList,np.mean(word_vec,axis = 0)])

            fout = open(outpath[i],'wb')
            pkl.dump(vec_aveList,fout)
            fout.close()

if __name__ =="__main__":
    argvs = sys.argv
    arg_length = len(argvs)
    if arg_length < 4 or arg_length > 6:
        print("コマンドライン引数の数が合いません。終了します。")
        quit()

    if arg_length == 4:
        result_process = 'w2v'
        Data_flag = (argvs[1]=='T')
        dict_flag = (argvs[2]=='T')
        model_flag = (argvs[3]=='T')

    else:
        result_process = argvs[1]
        Data_flag = (argvs[2]=='T')
        dict_flag = (argvs[3]=='T')
        model_flag = (argvs[4]=='T')

    #ファイル名をまとめているテキストからファイル名を読み出す
    f = open('file_name.txt','r')
    #ファイル名がすべて接続されて居るので区切り文字を指定してファイル名ごとに分ける
    lines = f.readlines()
    fn = [line.strip() for line in lines]
    dic_path = './data/dict.csv'
    #オブジェクト作成
    myData = makeData(fn,len(fn))
    w2v = word2vec_livedoor(myData.fn_out,myData.fn_model,dic_path)
    if Data_flag:
        for i in np.arange(myData.id):
            print("inputfile:{}".format(myData.fn_in[i]))
            print("outputfile:{}".format(myData.fn_out[i]))
            myData.xml_shaping(myData.fn_in[i],myData.fn_in_txt[i])
            myData.wakati(myData.fn_in_txt[i],myData.fn_out[i])

    if dict_flag:
        myData.make_dic(myData.fn_out,dic_path)

    if model_flag:
        w2v.w2v_train()

    if result_process == 'w2v':
        w2v.mean_w2v(myData.fn_w2v)

    elif result_process == 'tfidf':
        w2v.mean_w2v_tfidf(myData.fn_tfidf)
                        

