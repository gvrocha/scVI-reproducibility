from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
import scVI
from benchmarking import *
from helper import *

#---
indata_path  = "/scVI/data/processed/CORTEX/"
outdata_path = "/scVI/data/results/CORTEX/"

#---
print("%s: Defining functions" % datetime.now().strftime("%Y/%m/%d %H:%M:%S %Z%z"))
def mkdirp(path):
  if not os.path.exists(path):
    try:  
      os.makedirs(path)
    except OSError:  
      print ("Creation of the directory %s failed" % path)
    else:  
      print ("Successfully created the directory %s" % path)
  else:
    if not os.path.isdir(path):
      raise ValueError('%s exists and is NOT a directory' % (path))

def agg_clusters_2(c):
    new_c = np.zeros_like(c)
    for t in np.arange(c.shape[0]):
        if c[t] in [2, 6, 5]:
            new_c[t] = 0
        else:
            new_c[t] = 1
    return new_c

def agg_clusters_3(c):
    new_c = np.zeros_like(c)
    for t in np.arange(c.shape[0]):
        if c[t] in [2, 6, 5]:
            new_c[t] = 0
        elif c[t] in [1, 3]: 
            new_c[t] = 1
        else:
            new_c[t] = 2
    return new_c

def agg_clusters_4(c):
    new_c = np.zeros_like(c)
    for t in np.arange(c.shape[0]):
        if c[t] in [2, 6, 5]:
            new_c[t] = 0
        elif c[t] in [1, 3]: 
            new_c[t] = 1
        elif c[t] in [0]:
            new_c[t] = 2
        else:
            new_c[t] = 3
    return new_c

#---
print("%s: Setting parameters" % datetime.now().strftime("%Y/%m/%d %H:%M:%S %Z%z"))
learning_rate = 0.0004
epsilon = 0.01

#---
# expression data
print("%s: Reading input data" % datetime.now().strftime("%Y/%m/%d %H:%M:%S %Z%z"))
expression_train = np.loadtxt(indata_path + "data_train")
expression_test = np.loadtxt(indata_path+ "data_test")

# zero masked matrix
X_zero, i, j, ix = \
        np.load(indata_path + "imputation/X_zero.npy"), \
	np.load(indata_path + "imputation/i.npy"),\
        np.load(indata_path + "imputation/j.npy"), \
	np.load(indata_path + "imputation/ix.npy")

#labels
c_train = np.loadtxt(indata_path + "label_train")
c_test = np.loadtxt(indata_path + "label_test")

#---
expression_train.shape

#---
print("%s: Setting up TensorFlow model" % datetime.now().strftime("%Y/%m/%d %H:%M:%S %Z%z"))
tf.compat.v1.reset_default_graph()
expression     = tf.compat.v1.placeholder(tf.float32, (None, expression_train.shape[1]), name='x')
kl_scalar      = tf.compat.v1.placeholder(tf.float32, (), name='kl_scalar')
optimizer      = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
training_phase = tf.compat.v1.placeholder(tf.bool, (), name='training_phase')

# getting priors
log_library_size = np.log(np.sum(expression_train, axis=1))
mean, var        = np.mean(log_library_size), np.var(log_library_size)

# loading data
model = scVI.scVIModel(expression=expression, \
                       kl_scale=kl_scalar, \
                       optimize_algo=optimizer, \
                       phase=training_phase, \
                       library_size_mean=mean, \
                       library_size_var=var)

#---
#starting computing session
print("%s: Starting computing session" % datetime.now().strftime("%Y/%m/%d %H:%M:%S %Z%z"))
sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())
result = train_model(model, (expression_train, expression_test), sess, 250)

dic_full = {expression: expression_train, training_phase:False}
latent = sess.run(model.z, feed_dict=dic_full)
cluster_scores(latent, 7, c_train)

print(cluster_scores(latent, 2, agg_clusters_2(c_train)))
print(cluster_scores(latent, 3, agg_clusters_3(c_train)))
print(cluster_scores(latent, 4, agg_clusters_4(c_train)))
print(cluster_scores(latent, 7, c_train))
