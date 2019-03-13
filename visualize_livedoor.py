# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import pdb
import pickle
import numpy as np
import pandas as pd
from matplotlib import colors
from gensim.models import Word2Vec

# select method
method = 1

if method == 1:
	dataPath = 'livedoor-news-data-pkl/w2v'
	savedataPath = 'livedoor-news-data-TSNE/w2v'
	imgName = 'w2v_wiki.png'

elif method == 2:
	dataPath = 'livedoor-news-data-pkl/tfidf'
	savedataPath = 'livedoor-news-data-TSNE/tfidf'
	imgName = 'tfidf.png'



# ファイル名（カテゴリに対応）をまとめているテキストからファイル名を読み出す
f = open('file_name_4cate.txt','r')
# ファイル名がすべて接続されて居るので区切り文字を指定してファイル名ごとに分ける
lines = f.readlines()
fn = [line.strip() for line in lines]

# 各カテゴリのファイルごとにpickleファイルから特徴量を読み込み、XとYに格納
# X：特徴量
# Y: カテゴリID
flag = 0
for fInd in np.arange(len(fn)):
	fullDataPath = os.path.join(dataPath,'{}.pickle'.format(fn[fInd]))
	print(fullDataPath)
	with open(fullDataPath,'rb') as fp:
		tmpX = pickle.load(fp)
		tmpY = np.ones(tmpX.shape[0])*fInd

		if flag == 0:
			X = tmpX
			Y = tmpY
			flag = 1
		else:
			X = np.vstack([X,tmpX])
			Y = np.hstack([Y,tmpY])

data_num = X.shape[0]
dict = pd.read_csv('data/dict.csv')['word'].values
dict_ind = np.random.permutation(dict.shape[0])
dict_word = dict[dict_ind[:50]].tolist()
model = Word2Vec.load('livedoor-news-data-model_notwikipedia/word2vec.model')
dict_vec = model.wv[dict_word]
X = np.vstack([X,dict_vec])


# T-SNE
X2d = TSNE(n_components=2, random_state=0).fit_transform(X)
# colormap
lcmap = colors.ListedColormap(['#000000', '#FF99FF', '#8000FF',
                               '#0000FF', '#0080FF', '#58FAF4',
                               '#00FF00', '#FFFF00', '#FF8000',
                               '#FF0000'])

plt.scatter(X2d[:data_num, 0], X2d[:data_num, 1], c=Y+1, cmap=lcmap)
plt.colorbar()
plt.savefig(imgName)
plt.show()
fullsaveDataPath = os.path.join(savedataPath,'data_label.pickle')
fp = open(fullsaveDataPath,mode='wb')
pickle.dump(X2d,fp)
pickle.dump(Y,fp)
fp.close()
'''
for count in np.arange(len(fn)):
	fullsaveDataPath = os.path.join(savedataPath,'{}.pickle'.format(fn[count]))
	data = X2d[np.where(Y==count)]
	fp = open(fullsaveDataPath,mode='wb')
	pickle.dump(data,fp)
	fp.close()
'''
with open('dict_word_vec.pickle','wb') as fw:
    pickle.dump(dict_word,fw)
    pickle.dump(dict_vec,fw)
    pickle.dump(X2d[data_num:],fw)

