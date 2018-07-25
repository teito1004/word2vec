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
method = 2#1:w2v,2:tfidf
train_per = 0.8


def weight_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name,shape):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))

#tensorflowで用いるデータ群作成
input_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='input_data')
#input_label = tf.placeholder(shape=(None),dtype = tf.int32,name = 'input_label')
target_data = tf.placeholder(shape=(None,None),dtype = tf.float32,name='target_data')
#target_label = tf.placeholder(shape=(None),dtype = tf.int32,name = 'target_label')

#線形回帰で必要なW,Bを作成
W1 = weight_variable('weight1',[src_vec_length,hidden_unit_num])
B1 = weight_variable('bias1',[hidden_unit_num])
W2 = weight_variable('weight2',[hidden_unit_num,tar_vec_length])
B2 = weight_variable('bias2',[tar_vec_length])

#活性化関数を指定してfc層を生成。
fc1 = tf.nn.relu(tf.add(tf.matmul(input_data,W1),B1))
#fc1 = tf.nn.relu(tf.add(tf.matmul(input_data,W1),B1))
fc2 = tf.add(tf.matmul(fc1,W2),B2)
#fc2 = tf.nn.relu(tf.add(tf.matmul(fc1,W2),B2))

#loss関数(平均二乗誤差)
loss = tf.reduce_mean(tf.square(target_data - fc2))

#optimizerの設定
train_optimaizer = tf.train.AdamOptimizer(0.1).minimize(loss)

#初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#-------------------ここまでtensorflowの設定-----------------

#使用する手法ごとにデータパスを作成する。
if method == 1:
    src_dataPath = 'livedoor-news-data-TSNE/w2v/data_label.pickle'
    tar_dataPath = 'livedoor-news-data-pkl/w2v'

elif method == 2:
    src_dataPath = 'livedoor-news-data-TSNE/tfidf/data_label.pickle'
    tar_dataPath = 'livedoor-news-data-pkl/tfidf'

#srcデータとtarデータで保存方法が違うため、それぞれを専用に読み込む部分を作成する。

#T-SNEで２次元に削減したデータと正解ラベルのペアをpickleファイルから読み込む
fp = open(src_dataPath,'rb')
src_data = pkl.load(fp)
src_label = pkl.load(fp)
src_batch_size = src_data.shape[0]
#train用、test用で分割する。
src_train_data = src_data[:int(src_data.shape[0]*train_per)]
src_test_data = src_data[int(src_data.shape[0]*train_per):]
randInd_train = np.random.permutation(src_train_data.shape[0])
randInd_test = np.random.permutation(src_test_data.shape[0])
fp.close()

#100次元のベクトルデータを読み込む
# ファイル名（カテゴリに対応）をまとめているテキストからファイル名を読み出す
f = open('file_name.txt','r')
# ファイル名がすべて接続されて居るので区切り文字を指定してファイル名ごとに分ける
lines = f.readlines()
fn = [line.strip() for line in lines]
flag = 0
for fInd in np.arange(len(fn)):
	tar_fulldataPath = os.path.join(tar_dataPath,'{}.pickle'.format(fn[fInd]))
	#print(tar_fulldataPath)
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
#train用、test用で分割する。
tar_train_data = tar_data[:int(tar_data.shape[0]*train_per)]
tar_test_data = tar_data[int(tar_data.shape[0]*train_per):]



#batch作成
def next_batch_train(count,Ind):
    #epochが終了しているならばIndexを作り直す
    if(src_train_data.shape[0] < (batch_size*(count+1))):
        Ind = np.random.permutation(src_train_data.shape[0])
        count = 0
    #batch_size分だけ取り出して、カウントを増やす
    batch_src = src_train_data[Ind[batch_size*count:batch_size*(count+1)]]
    batch_tar = tar_train_data[Ind[batch_size*count:batch_size*(count+1)]]
    #辞書型にして返す
    return {
        input_data:batch_src,
        target_data:batch_tar,
    },count+1,Ind

def next_batch_test(count,Ind):
    #epochが終了しているならばIndexを作り直す
    if(src_test_data.shape[0] < (batch_size*(count+1))):
        Ind = np.random.permutation(src_test_data.shape[0])
        count = 0
    #batch_size分だけ取り出して、カウントを増やす
    batch_src = src_test_data[Ind[batch_size*count:batch_size*(count+1)]]
    batch_tar = tar_test_data[Ind[batch_size*count:batch_size*(count+1)]]
    #辞書型にして返す
    return {
        input_data:batch_src,
        target_data:batch_tar,
    },count+1,Ind

#過程のsave用に必要なクラスを宣言
saver=tf.train.Saver()
#学習の開始
for step in np.arange(ite+1):
    fd,train_count,randInd_train = next_batch_train(train_count,randInd_train)
    step_loss = str(sess.run(loss,feed_dict=fd))
    if step%100 == 0:
        print('step:{},loss:{}'.format(step,step_loss))
        saver.save(sess,'restration_model/model.ckpt',global_step=step)

    if step%500 ==0:
        fd,test_count,randInd_test = next_batch_test(test_count,randInd_test)
        step_loss,predicted = sess.run([loss,fc2],feed_dict=fd)
        print('---------test_sequence---------')
        print('loss:{}'.format(str(step_loss)))
        print('input ->:{}'.format(fd[input_data][0]))
        print('predicted ->:{}'.format(str(predicted[0])))
        print('-------------------------------')
