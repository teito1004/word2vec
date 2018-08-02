import pdb
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pkl
import os

batch_size = 100
train_count = 0
test_count = 0
ite = 10000
src_vec_length = 2
hidden_unit_num = 200
tar_vec_length = 100
method = 1	#1:w2v,2:tfidf
train_per = 0.8

dist_mode = 1
dist_fname = 'distance.pkl'
nNearData = 50

#===========================
# レイヤーの関数
# fc layer
def fc_relu(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	fc = tf.nn.relu(fc)
	return fc
	
# fc layer
def fc(inputs, w, b):
	fc = tf.matmul(inputs, w) + b
	return fc	

def weight_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
#===========================

#===========================
# tensorflowで用いるデータ群作成
input_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='input_data')
target_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='target_data')

# 線形回帰で必要なW,Bを作成
W1 = weight_variable('weight1',[src_vec_length*nNearData,hidden_unit_num])
B1 = weight_variable('bias1',[hidden_unit_num])
W2 = weight_variable('weight2',[hidden_unit_num,hidden_unit_num])
B2 = weight_variable('bias2',[hidden_unit_num])
W3 = weight_variable('weight3',[hidden_unit_num,tar_vec_length])
B3 = weight_variable('bias3',[tar_vec_length])

# 活性化関数を指定してfc層を生成。
fc1 = fc_relu(input_data, W1, B1)
fc2 = fc_relu(fc1, W2, B2)
fc_out = fc(fc2,W3,B3)
#===========================

#===========================
# loss関数(平均二乗誤差)
loss = tf.reduce_mean(tf.abs(target_data - fc_out))

# optimizerの設定
train_optimaizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# 初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#===========================

#===========================
# 使用する手法ごとにデータパスを作成する。
if method == 1:
    src_dataPath = 'livedoor-news-data-TSNE/w2v/data_label.pickle'
    tar_dataPath = 'livedoor-news-data-pkl/w2v'

elif method == 2:
    src_dataPath = 'livedoor-news-data-TSNE/tfidf/data_label.pickle'
    tar_dataPath = 'livedoor-news-data-pkl/tfidf'
#===========================


#===========================
# T-SNEで２次元に削減したデータと正解ラベルのペアをpickleファイルから読み込む
with open(src_dataPath,'rb') as fp:
	src_data = pkl.load(fp)

#---------
# T-SNE2点間の距離の計算
if dist_mode == 1:
	# データ点間距離の計算
	src_data1=np.tile(src_data,(src_data.shape[0],1,1))
	src_data2 = np.transpose(src_data1,[1,0,2])
	src_dist = np.sum(np.square(src_data1-src_data2),axis=2)
	
	with open(dist_fname,'wb') as fp:
		pkl.dump(src_dist,fp)
			
elif dist_mode == 2:
	with open(dist_fname,'rb') as fp:
		src_dist = pkl.load(fp)
#---------

#---------
# nNearData近傍のデータを集めて一つのベクトルにする
nearInds = np.argsort(src_dist,axis=1)[:,:nNearData]
src_data_near = np.reshape(src_data[nearInds,:],[-1,src_vec_length * nNearData])
#---------

# train用、test用で分割する。
randInd_all = np.random.permutation(src_data_near.shape[0])
src_train_data = src_data_near[randInd_all[:int(src_data_near.shape[0]*train_per)]]
src_test_data = src_data_near[randInd_all[int(src_data_near.shape[0]*train_per):]]
randInd_train = np.random.permutation(src_train_data.shape[0])
#===========================


#===========================
# word2vecの100次元のベクトルデータを読み込む
# ファイル名（カテゴリに対応）をまとめているテキストからファイル名を読み出す
f = open('file_name.txt','r')
# ファイル名がすべて接続されて居るので区切り文字を指定してファイル名ごとに分ける
lines = f.readlines()
fn = [line.strip() for line in lines]
flag = 0
for fInd in np.arange(len(fn)):
	tar_fulldataPath = os.path.join(tar_dataPath,'{}.pickle'.format(fn[fInd]))
	with open(tar_fulldataPath,'rb') as fp:
		tmpX = pkl.load(fp)
		tmpY = np.ones(tmpX.shape[0])*fInd

		if flag == 0:
			tar_data = tmpX
			tar_label = tmpY
			flag = 1
		else:
			tar_data = np.vstack([tar_data,tmpX])
			tar_label = np.hstack([tar_label,tmpY])

#---------
# word2vec特徴量2点間の距離の計算
if dist_mode == 3:
	# データ点間距離の計算
	tar_data1=np.tile(tar_data,(tar_data.shape[0],1,1))
	tar_data2 = np.transpose(tar_data1,[1,0,2])
	tar_dist = np.sum(np.square(tar_data1-tar_data2),axis=2)
	
	with open(dist_fname,'wb') as fp:
		pkl.dump(src_dist,fp)
		pkl.dump(tar_dist,fp)
#---------

# train用、test用で分割する。
# 標準偏差が1になるように正規化
tar_data = tar_data/np.tile(np.std(tar_data,axis=0),[tar_data.shape[0],1])

tar_train_data = tar_data[randInd_all[:int(tar_data.shape[0]*train_per)]]
tar_test_data = tar_data[randInd_all[int(tar_data.shape[0]*train_per):]]
#===========================

#===========================
# batch作成
def next_batch_train(count,Ind):
    #epochが終了しているならばIndexを作り直す
    if(src_train_data.shape[0] < (batch_size*(count+1))):
        Ind = np.random.permutation(src_train_data.shape[0])
        count = 0
    
    # batch_size分だけ取り出して、カウントを増やす
    batch_src = src_train_data[Ind[batch_size*count:batch_size*(count+1)]]
    batch_tar = tar_train_data[Ind[batch_size*count:batch_size*(count+1)]]
    
    # 辞書型にして返す
    return {
        input_data:batch_src,
        target_data:batch_tar,
    },count+1,Ind

#===========================

#===========================
# 過程のsave用に必要なクラスを宣言
saver=tf.train.Saver()
# 学習の開始
for step in np.arange(ite+1):
    fd,train_count,randInd_train = next_batch_train(train_count,randInd_train)
    _, step_loss = sess.run([train_optimaizer, loss],feed_dict=fd)
    if step%1 == 0:
        print('step:{},loss:{}'.format(step,str(step_loss)))
        saver.save(sess,'restration_model/model.ckpt',global_step=step)

    if step%100 ==0:
        step_loss,predicted = sess.run([loss,fc_out],feed_dict={input_data:src_test_data, target_data: tar_test_data})
        print('---------test_sequence---------')
        print('loss:{}'.format(str(step_loss)))
        print('input ->:{}'.format(fd[input_data][0]))
        print('predicted ->:{}'.format(str(predicted[0])))
        print('gt ->:{}'.format(str(tar_test_data[0])))
        print('-------------------------------')
#===========================