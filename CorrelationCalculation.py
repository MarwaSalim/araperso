import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("DataSet_11_Nov_Tweets_csv1.csv")
new_df=df.drop(['ProfilePic','ID'], axis=1).corr(method='spearman')
new_df.to_csv('CorrelationSpearman11Nov.csv')
new_df=df.drop(['ProfilePic','ID'], axis=1).corr(method='pearson')
new_df.to_csv('CorrelationPearson11Nov.csv')
new_df=df.drop(['ProfilePic','ID'], axis=1).corr(method='kendall')
new_df.to_csv('Correlationkendall11Nov.csv')
# plt.rcParams['figure.figsize'] = (50,50)
#
# fig, ax = plt.subplots(nrows=1, ncols=1)
#
# #ax=ax.flatten()
# cols = [
#     'NumOfFollowers','NumOfFollowing','NumOfTweetsPerDay',
#         #'O','C','E','A','N',
# ]
# colors=['#415952', '#f35134', '#243AB5', '#243AB5','#415952', '#f35134', '#243AB5', '#243AB5']
#
# bx=['O_Sc','C','E','A','N']
# j=0
# x=0
# index=0
# #for i in ax:
# if j%1 ==0:
#         x=j/1
#         index=0
#         ax.set_ylabel(bx[x])
# ax.scatter(df[cols[index]],
#               df[bx[x]],
#               alpha=0.5,
#               color=colors[index])
# ax.set_xlabel(cols[index])
# ax.set_title('Pearson: %s'%df.corr().loc[cols[index]][bx[x]].round(2)+' Spearman: %s'%df.corr(method='spearman').loc[cols[index]][bx[x]].round(2))
# j+=1
# index+=1
#
# plt.show()