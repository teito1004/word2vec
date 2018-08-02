# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import pdb
import pickle
import numpy as np
from matplotlib import colors

# select method
method = 1

if method == 1:
	dataPath = 'livedoor-news-data-pkl/w2v'
	savedataPath = 'livedoor-news-data-TSNE/w2v'
	imgName = 'w2v.png'

elif method == 2:
	dataPath = 'livedoor-news-data-pkl/tfidf'
	savedataPath = 'livedoor-news-data-TSNE/tfidf'
	imgName = 'tfidf.png'



# ファイル名（カテゴリに対応）をまとめているテキストからファイル名を読み出す
f = open('file_name.txt','r')
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

# T-SNE
X2d = TSNE(n_components=2, random_state=0).fit_transform(X)
# colormap
lcmap = colors.ListedColormap(['#000000', '#FF99FF', '#8000FF',
                               '#0000FF', '#0080FF', '#58FAF4',
                               '#00FF00', '#FFFF00', '#FF8000',
                               '#FF0000'])

plt.scatter(X2d[:, 0], X2d[:, 1], c=Y+1, cmap=lcmap)
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
