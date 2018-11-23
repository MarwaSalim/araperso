import pandas as pd
import MNB
import MNB_OverSampling
import MNB_UnderSampling
import SVM
import SVM_OverSampling
import SVM_UnderSampling
import DT
import DT_OverSampling
import DT_UnderSampling
import KNN
import KNN_OverSampling
import KNN_UnderSampling
# from sklearn.model_selection import train_test_split
# df = pd.read_csv("DataSet_11_Nov_Tweets_csv1.csv")
# trainSamples, testSamples = train_test_split(df, test_size=0.34)
# testSamples.to_csv("TestSampling.csv")
# trainSamples.to_csv("TrainSampling.csv")
# df=trainSamples
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=1)
# #UNBALANCE
# df.O.value_counts().plot(kind='bar', title='Count (O)')
# print "O"
# print df.O.value_counts()
# plt.show()
# df.C.value_counts().plot(kind='bar', title='Count (C)')
# print "C"
# print df.C.value_counts()
# plt.show()
# df.E.value_counts().plot(kind='bar', title='Count (E)')
# print "E"
# print df.E.value_counts()
# plt.show()
# #UNBALANCE
# df.A.value_counts().plot(kind='bar', title='Count (A)')
# print "A"
# print df.A.value_counts()
# plt.show()
# df.N.value_counts().plot(kind='bar', title='Count (N)')
# print "N"
# print df.N.value_counts()
# plt.show()

trainSamples = pd.read_csv("DataSet_11_Nov_Tweets_csv1.csv")
# MNB.Begin(trainSamples,"O")
# MNB.Begin(trainSamples,"C")
# MNB.Begin(trainSamples,"E")
# MNB.Begin(trainSamples,"A")
# MNB.Begin(trainSamples,"N")
# MNB_OverSampling.Begin(trainSamples,"O")
#ValueError: Expected n_neighbors <= n_samples,  but n_samples = 5, n_neighbors = 6
MNB_OverSampling.Begin(trainSamples,"A")
# MNB_UnderSampling.Begin(trainSamples,"O")
# MNB_UnderSampling.Begin(trainSamples,"A")
#
# SVM.Begin(trainSamples,"O")
# SVM.Begin(trainSamples,"C")
# SVM.Begin(trainSamples,"E")
# SVM.Begin(trainSamples,"A")
# SVM.Begin(trainSamples,"N")
SVM_UnderSampling.Begin(trainSamples,"O")
SVM_UnderSampling.Begin(trainSamples,"A")
SVM_OverSampling.Begin(trainSamples,"O")
SVM_OverSampling.Begin(trainSamples,"A")

# DT.Begin(trainSamples,"O")
# DT.Begin(trainSamples,"C")
# DT.Begin(trainSamples,"E")
# DT.Begin(trainSamples,"A")
# DT.Begin(trainSamples,"N")
# DT_UnderSampling.Begin(trainSamples,"O")
# DT_UnderSampling.Begin(trainSamples,"A")
# DT_OverSampling.Begin(trainSamples,"O")
#ValueError: Expected n_neighbors <= n_samples,  but n_samples = 5, n_neighbors = 6
DT_OverSampling.Begin(trainSamples,"A")

# KNN.Begin(trainSamples,"O")
# KNN.Begin(trainSamples,"C")
# KNN.Begin(trainSamples,"E")
# KNN.Begin(trainSamples,"A")
# KNN.Begin(trainSamples,"N")
# KNN_UnderSampling.Begin(trainSamples,"O")
# KNN_UnderSampling.Begin(trainSamples,"A")
# KNN_OverSampling.Begin(trainSamples,"O")
KNN_OverSampling.Begin(trainSamples,"A")