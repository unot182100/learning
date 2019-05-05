# -*- coding utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import seaborn as sns
#dropbox

onehot_col = ['v0','v1']
#onehot_col = ['v0','v1','v2','v3','v4','v5','v6','v7']
res_num = len(onehot_col)
features = ['ssc', 'sBVP', 'srsp', 'nirs', 'utterance' ]
#features = ['tsc', 'tBVP', 'trsp', 'nirs', 'ssc', 'sBVP', 'srsp', 'utterance' ]

feat_num = len(features)

cv_loss_min = 100

def load_data():
  label_col = ['semotion']

  # ファイルパスは各々変えてください
  all_data =  pd.read_csv('newnew.csv',delimiter=',')
  #データ読み込み
  feature_data = all_data[features]
  #データの切り離し
  label_data  = all_data[label_col]
  #データの切り離し
  onehot_exp = pd.get_dummies(label_data['semotion'])
  #  4      だったのを、0  0  0  1  0みたいにする
  onehot_exp.columns = onehot_col
  #カラム名を付ける
  data = pd.concat([feature_data,label_data,onehot_exp],axis=1)
  #すべてのデータをくっつける
  data = data/np.max(data)
  #標準化？？してんのかな

  return data


data = load_data()
# data is separated 60:40
np.random.seed(1000)
#発生する乱数をあらかじめ固定(100)でもいいの？？

N = 60
train_num = int(data.shape[0]*N/100)
#dataが(134120, 14)で、それの134120のうち60%を選んだってこと
indexes = np.random.choice(data.shape[0],train_num, replace=False)
#ランダムにdata.shapeの134120から60%の80472個くらいを重複なく選ぶ。seedしてるので毎回同じ



train = data.sample(frac=N/100) # train data ( 60 percent of data14 )
#ん、なんかまた選び直してる？??????????????

test  = data.loc[~data.index.isin(train.index)] # test data
# ~でビット反転 まあ、40%を抽出してる。trainとかぶらないようにしているのか。
x_dataa = train[features]
#trainの中のfeatursのカラムだけ60%とる
#              ssc      sBVP      srsp      nirs  utterance
# 0       0.473533 -0.338842  0.903669  0.432346   0.333333
# 2       0.473382 -0.535421  0.903872  0.428180   0.333333
# 5       0.473382 -0.570382  0.903867  0.421931   0.333333
# 11      0.473230 -0.648648  0.903861  0.400547   0.333333
# ,,,,,
# 134118  0.704174 -0.699927  0.906551 -1.755336   1.000000
# 134119  0.704174 -0.687815  0.906754 -1.756288   1.000000

# [53648 rows x 5 columns]

y_dataa = train[onehot_col]
#trainの中の心的状態のカラムだけ60%とる
x_datab = test[features]
#テストデータのカラムから40%とる
y_datab = test[onehot_col]
#テストデータのカラムの心的状態のカラムから40%とる



with tf.Graph().as_default():
  #TensorBoardのグラフに出力するスコープを指定
  tf.set_random_seed(1000)
  inp = feat_num
  hidden1 = 43
  hidden2 = 31
  hidden3 = 29
  hidden4 = 21
  output = 2
  alpha = .02

  # Network ========================================================================
  x = tf.placeholder(tf.float32, shape=[None,inp])
  #入力データの次元のセット。入力データはinput次元なので、セット。Noneの方は入力データをどれくらい与えるかなので、まだ決めてませんってこと。
  y = tf.placeholder(tf.float32, shape=[None,output])
#正解ラベルの次元をセット！

  W_fc1 = tf.Variable(tf.truncated_normal([inp,hidden1],mean=0.0, stddev=1.0))
  b_fc1 = tf.Variable(tf.constant(0.0,shape=[hidden1]))
  z_fc1 = tf.matmul(x,W_fc1)+b_fc1
  h_fc1 = tf.tanh(z_fc1)
  #Variableは、重み行列とバイアスの宣言
  #入力層と中間層の間

#層を増やす場合はここから下をいじる.
  W_fc2 = tf.Variable(tf.truncated_normal([hidden1,hidden2],mean=0.0, stddev=1.0))
  b_fc2 = tf.Variable(tf.constant(0.0,shape=[hidden2]))
  z_fc2 = tf.matmul(h_fc1,W_fc2)+b_fc2
  h_fc2 = tf.tanh(z_fc2)

  W_fc3 = tf.Variable(tf.truncated_normal([hidden2,hidden3],mean=0.0, stddev=1.0))
  b_fc3 = tf.Variable(tf.constant(0.0,shape=[hidden3]))
  z_fc3 = tf.matmul(h_fc2,W_fc3)+b_fc3
  h_fc3 = tf.tanh(z_fc3)

  W_fc4 = tf.Variable(tf.truncated_normal([hidden3,hidden4],mean=0.0, stddev=1.0))
  b_fc4 = tf.Variable(tf.constant(0.0,shape=[hidden4]))
  z_fc4 = tf.matmul(h_fc3,W_fc4)+b_fc4
  h_fc4 = tf.tanh(z_fc4)

  W_out = tf.Variable(tf.truncated_normal([hidden4,output],mean=0.0,stddev=1.0))
  b_out = tf.Variable(tf.constant(0.0,shape=[output]))
  out_y = tf.matmul(h_fc4,W_out)+b_out
  #出力そう
  #=================================================================================
  #saverのやつ
  # saver = tf.train.Saver()
  # saver.restore(sess, "model.ckpt")

  sm_ce = tf.nn.softmax_cross_entropy_with_logits(logits=out_y,labels=y)
  #ソフトマックス設定
  loss = tf.reduce_mean(sm_ce)
  #損失関数の定義

  #学習方法の決定
  opt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
  #tf.train.GradientDescentOptimizer(alpha)
  #オプティマイザー設定。最適化法の設定 AdamとかAdeDeltaとかある
  #minimize(loss)
  #最小化すべき値を設定

  correct_prediction = tf.equal(tf.argmax(out_y,1),tf.argmax(y,1))
  #このコードにより、ニューラルネットの出力ベクトルで最も大きかった成分の番号（つまり判定されたクラス）と、正解ラベルで最も大きな成分の番号（正解クラス）が等しいか否かを調べます。
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #によって、出力と正解を調べた結果を獲得します。
#例えば[True,True,False,True]ならば[1,1,0,1]となり、平均が3/4で0.75と正解率が出ます。


#以下のグタツで初期化の実行
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  #sess.run(tf.initialize_all_variables())
  #初期化 Tensolfloeではニューラルネットの学習を始める前にこの操作が必要です（じゃないと動かないそうです）。


  train_loss_series = []
  train_acc_series = []

  cv_loss_series = []
  cv_acc_series = []
  #saveのやつ
# Training
  for i in range(70000):
#10000回繰り替えす

    train_loss, train_acc, _ = sess.run([loss,accuracy,opt], feed_dict={x: x_dataa, y: y_dataa})
    cv_loss, cv_acc = sess.run([loss,accuracy], feed_dict={x: x_datab, y: y_datab})

    train_loss_series.append(train_loss)
    #.appendでリストオブジェクトの最後に引数を追加
    #できあがったのを順々に入れてる
    train_acc_series.append(train_acc)
    #できあがったのを順々に入れてる
    cv_loss_series.append(cv_loss)
    #できあがったのを順々に入れてる
    cv_acc_series.append(cv_acc)
    #できあがったのを順々に入れてる


    if(i%1000==0):
      print("Epoch:[%d] | loss %1.4f | acc %1.4f" % (i, train_loss, train_acc))
#%1.4fで、1.0000  %1.8fで1.00000000みたいになる
      print("cvloss %1.4f | cvacc %1.4f" % (cv_loss,cv_acc))

#saver.save(sess, "model.ckpt", global_step=100)

#プロット
  plt.subplot(2, 1, 1)
  #並べたいと時に
#plt.subplot(行数, 列数, 何番目のプロットか)
  plt.plot(range(len(train_loss_series)), train_loss_series, linewidth=2)
  plt.plot(range(len(train_acc_series)), train_acc_series, linewidth=2)
  #横軸のスタートはtrain_loss_series,終わりはtrain_acc_series
  #縦軸のスタートはtrain_loss_series,終わりはtrain_acc_series
  plt.title('Training')
  plt.xlabel('step')
  plt.legend(["train_loss","train_acc"],loc='best')
  #右上に出るいい感じのやつ。

  plt.subplot(2, 1, 2)
  #並べたいと時に
#plt.subplot(行数, 列数, 何番目のプロットか)
  plt.plot(range(len(cv_loss_series)), cv_loss_series, linewidth=2)
  plt.plot(range(len(cv_acc_series)), cv_acc_series, linewidth=2)
  plt.title('Cross Validation')
  plt.xlabel('step')
  plt.legend(["cv_loss","cv_acc"],loc='best')

  plt.tight_layout()
  plt.show()