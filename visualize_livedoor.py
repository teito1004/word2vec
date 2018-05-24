# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import pdb
import pickle
import numpy as np

dataPath = 'livedoor-news-pkl'
    
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

X2d = TSNE(n_components=2, random_state=0).fit_transform(X)

plt.scatter(X2d[:, 0], X2d[:, 1], c=Y)
plt.colorbar()
plt.savefig('t-sne.png')
plt.show()

