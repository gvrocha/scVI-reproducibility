from datetime import datetime
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../../')
from benchmarking import *

#---
indata_path  = "/scVI/data/original/CORTEX/"
outdata_path = "/scVI/data/processed/CORTEX/"

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
            new_c[t] = 1
        else:
            new_c[t] = 0
    return new_c

def agg_clusters_3(c):
    new_c = np.zeros_like(c)
    for t in np.arange(c.shape[0]):
        if c[t] in [2, 6, 5]:
            new_c[t] = 2
        elif c[t] in [1, 3]: 
            new_c[t] = 1
        else:
            new_c[t] = 0
    return new_c

def agg_clusters_4(c):
    new_c = np.zeros_like(c)
    for t in np.arange(c.shape[0]):
        if c[t] in [2, 6, 5]:
            new_c[t] = 3
        elif c[t] in [1, 3]: 
            new_c[t] = 2
        elif c[t] in [0]:
            new_c[t] = 1
        else:
            new_c[t] = 0
    return new_c

#---
mkdirp(outdata_path)
mkdirp(outdata_path + "imputation")

#---
X = pd.read_csv(indata_path + "expression_mRNA_17-Aug-2014.txt", sep="\t", low_memory=False).T

#---
clusters = np.array(X[7], dtype=str)[2:]
precise_clusters = np.array(X[0], dtype=str)[2:]
celltypes, labels = np.unique(clusters, return_inverse=True)
_, precise_labels = np.unique(precise_clusters, return_inverse=True)
gene_names = np.array(X.iloc[0], dtype=str)[10:]
X = X.loc[:, 10:]
X = X.drop(X.index[0])
expression = np.array(X, dtype=np.int)[1:]

print(expression.shape[0], " cells with ", expression.shape[1], " genes")

#---
for i in np.unique(labels):
    print(i, clusters[np.where(labels == i)[0][0]], "\t" , len( np.where(labels == i)[0]))

#---
np.save(outdata_path + "labels", labels)
np.save(outdata_path + "labels_2", agg_clusters_2(labels))
np.save(outdata_path + "labels_3", agg_clusters_3(labels))
np.save(outdata_path + "labels_4", agg_clusters_4(labels))

#---
selected = np.std(expression, axis=0).argsort()[-558:][::-1]
expression = expression[:, selected]
gene_names = gene_names[selected].astype(str)

#---
#train test split for log-likelihood scores
X_train, X_test, c_train, c_test, cp_train, cp_test = train_test_split(expression, labels, precise_labels,\
                                                                       random_state=0)

#---
np.savetxt(outdata_path + "gene_expression", expression)
np.savetxt(outdata_path + "gene_name", gene_names, fmt="%s")
np.savetxt(outdata_path + "labels", labels)
np.savetxt(outdata_path + "precise_labels", precise_labels)

#---
np.savetxt(outdata_path + "data_train", X_train)
np.savetxt(outdata_path + "data_test", X_test)
np.savetxt(outdata_path + "label_train", c_train)
np.savetxt(outdata_path + "label_test", c_test)
np.savetxt(outdata_path + "precise_label_train", c_train)
np.savetxt(outdata_path + "precise_label_test", c_test)

#---
X_zero, i, j, ix = dropout(X_train)
np.save(outdata_path + "imputation/X_zero.npy", X_zero)
np.save(outdata_path + "imputation/i.npy", i)
np.save(outdata_path + "imputation/j.npy", j)
np.save(outdata_path + "imputation/ix.npy", ix)
