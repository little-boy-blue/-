#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from sklearn.datasets import load_iris
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()


# In[ ]:


iris = load_iris() #the dataset has 150 labeled datas  
def generate_data():
    for i,j in zip(iris["data"],iris["target"]):
        yield (i,j)   #i  is feature:[1.2, 3.1, 0.3, 6.2 ] ，j is label ：0 or 1 or 2
 
data = tf.data.Dataset.from_generator(generate_data,(tf.float32,tf.int32), (tf.TensorShape([4]), tf.TensorShape([])))
data = data.shuffle(150)# shuffle all data
data = data.batch(10)# 
#data = data.repeat(2)
data = data.repeat()
iter_data = data.make_one_shot_iterator()


# In[ ]:


#去一个数据出来看看
feature_e,label_e = iter_data.get_next()


# In[ ]:


#模型构建使用tf.keras API
#构建三层，每层的神经元个数分别为10，10，3的神经网络
model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
])


# In[ ]:


model(feature_e)


# In[ ]:


#定义损失函数，自动求梯度函数
#通过loss反向传播和梯度下降优化模型
#定义了loss 函数和自动求梯度函数。
def loss(model,x,y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# In[ ]:


#定义梯度下降优化器
#定义了一个学习率为0.01的优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step=tf.train.get_or_create_global_step()
loss_value, grads = grad(model,data_e,label_e)
optimizer.apply_gradients(zip(grads, model.variables),global_step)
print("Step: {},Loss: {}".format(global_step.numpy(),loss_value.numpy()))


# In[ ]:


#定义模型训练函数
def train_model(training_dataset, model, optimizer):
    train_loss_results = []
    train_accuracy_results = []
    for epoch in range(202):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()  
        for round_num in range(15):
                feature,label =  training_dataset.get_next() 
                loss , gradients = grad(model,feature,label)
                optimizer.apply_gradients(zip(gradients, model.variables),
                                          global_step=tf.train.get_or_create_global_step())
                epoch_loss_avg(loss)  # add current batch loss
                epoch_accuracy(tf.argmax(model(data_), axis=1, output_type=tf.int32),label)
        if epoch  % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


# In[ ]:


train_model(iter_ ,model, optimizer)


# In[ ]:


print("Prediction: {}".format(tf.argmax(model(feature_e), axis=1)))
print("Labels: {}".format(label_e))


# In[ ]:




