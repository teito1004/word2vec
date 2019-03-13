import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import pickle as pkl
import numpy as np
import pdb

with open('result_loss_w2v.pickle','rb') as fn:
    train_loss = pkl.load(fn)
    test_loss = pkl.load(fn)

with open('result_loss_w2v_4cate.pickle','rb') as fn:
    train_loss_2layer = pkl.load(fn)
    test_loss_2layer = pkl.load(fn)
'''
with open('result_loss_w2v_4layer.pickle','rb') as fn:
    train_loss_4layer = pkl.load(fn)
    test_loss_4layer = pkl.load(fn)

with open('result_loss_w2v_nearlest.pickle','rb') as fn:
    train_loss_nearlest = pkl.load(fn)
    test_loss_nearlest = pkl.load(fn)
'''
x = np.arange(0,10001,100)
plt.plot(x,train_loss,color = 'tomato' ,label='train_loss')
plt.plot(x,test_loss,color = 'cyan' ,label='test_loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.savefig('result_loss_w2v.png')
plt.show()
plt.cla()
plt.clf()

plt.plot(x,train_loss_2layer,color = 'tomato' ,label='train_loss')
plt.plot(x,test_loss_2layer,color = 'cyan' ,label='test_loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.savefig('result_loss_w2v_4cate.png')
plt.show()
plt.cla()
plt.clf()
'''
plt.plot(x,train_loss_4layer,color = 'tomato' ,label='train_loss_5layer')
plt.plot(x,test_loss_4layer,color = 'cyan' ,label='test_loss_5layer')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.savefig('result_loss_4layer.png')
plt.show()
plt.cla()
plt.clf()
'''
plt.plot(x,train_loss_2layer,color = 'tomato' ,label='train_loss_4cate')
plt.plot(x,test_loss_2layer,color = 'lightblue' ,label='test_loss_4cate')
plt.plot(x,train_loss,color = 'red' ,label='train_loss_9cate')
plt.plot(x,test_loss,color = 'cyan' ,label='test_loss_9cate')
#plt.plot(x,train_loss_4layer,color = 'darksalmon' ,label='train_loss_5layer')
#plt.plot(x,test_loss_4layer,color = 'deepskyblue' ,label='test_loss_5layer')
#plt.plot(x,train_loss_nearlest,color = 'sienna',label='train_loss_nearlest')
#plt.plot(x,test_loss_nearlest,color = 'teal' ,label='test_loss_nearlest')
plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('result_loss_w2v_all.png')
plt.show()
