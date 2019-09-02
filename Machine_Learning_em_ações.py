#!/usr/bin/env python
# coding: utf-8

# In[1]:


# bibliotecas importadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# The dataset contains data from every bovespa's stock of every useful day starting from 1998 to the end of 2018.
# https://www.kaggle.com/friedliver/bovespa
bov = pd.read_csv('../datasets/bovespa1998-2019.csv',low_memory=False)


# # Análise Exploratória

# In[3]:


bov.head()


# In[4]:


# Features
bov.columns


# # Significado das Colunas:<br>
# **TIPREG**: Registry type<br>
# **DATPRG**: Trading day<br>
# **CODBDI**: BDI code<br>
# **CODNEG**: Stock negotiation code<br>
# **TPMERC**: Market type<br>
# **NOMRES**: Short name of the stock issuing company<br>
# **ESPECI**: Stock specification<br>
# **PRAZOT**: Term in days of the market<br>
# **MODREF**: Reference currency<br>
# **PREABE**: Stock price on stock exchange opening<br>
# **PREMAX**: Maximum stock price during trading day<br>
# **PREMIN**: Minimum stock price during trading day<br>
# **PREMED**: Average stock price during trading day<br>
# **PREULT**: Last negotiation price of the stock during trading day<br>
# **PREOFC**: Best buying offer price of the stock during trading day<br>
# **PREOFV**: Best selling offer price of the stock during trading day<br>
# **TOTNEG**: Number of negotiations done regarding the stock during trading day<br>
# **QUATOT**: Total quantity of "títulos" negotiated<br>
# **VOLTOT**: Total volume of "títulos" negotiated<br>
# **PREEXE**: Stock option price<br>
# **INDOPC**: Correction indicator<br>
# **DATVEN**: Expiring date for stock options<br>
# **FATCOT**: Quotation factor<br>
# **PTOEXE**: Price in points<br>
# **CODISI**: Stock code<br>
# **DISMES**: Distribution number of the stock<br>

# In[5]:


bov.info()


# In[6]:


# Dados Ausentes
bov.isna().sum()


# In[7]:


# Entendendo a coluna PRAZOT
bov['PRAZOT'].value_counts()


# In[8]:


#Petroleo Brasileiro SA Petrobras
#BVMF: PETR3

bov.loc[(bov['CODNEG']=='PETR3')]


# In[9]:


# Reduzindo o dataset para somente algumas ações
sec = ['ABEV3','PETR3','VALE3','CIEL3','PSSA3']
bov = bov[bov['CODNEG'].isin(sec)]


# In[10]:


# Transformando a coluna de data para o formato certo
pd.to_datetime(bov['DATPRG'])


# In[11]:


bov.head()


# In[12]:


# Tipo das ações
bov['ESPECI'].value_counts()


# In[13]:


# Removendo colunas
temp = bov.drop(['TIPREG', 'CODBDI', 'TPMERC', 'TOTNEG', 'QUATOT', 'VOLTOT', 'INDOPC', 'DATVEN',
                 'FATCOT', 'PTOEXE', 'CODISI', 'DISMES'], axis=1)


# In[14]:


df = temp.copy()


# In[15]:


df.head()


# In[16]:


# Selecionando o período
start_date='2018-01-01'
end_date='2018-12-28'
mask = (df['DATPRG'] > start_date) & (df['DATPRG'] <= end_date)


# In[17]:


df = df.loc[mask]
df.head()


# In[18]:


# Gráfico com a evolução dos preços das ações em 2018
fig, ax = plt.subplots(figsize=(20,8))
sns.set(style='darkgrid',palette='dark',rc={"lines.linewidth":3})
sns.lineplot(x='DATPRG',y='PREABE',hue='CODNEG',data=df);


# # Modelos

# In[19]:


from sklearn.model_selection import cross_val_score
from sklearn import tree, svm, neighbors, ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[20]:


# Calculando o retorno logaritmico
df['logabe'] = np.log(df['PREABE'])
df['logmin'] = np.log(df['PREMIN'])
df['change'] = df['logabe'].subtract(df['logmin'])


# In[21]:


# Pivot Table com as ações
change = df.pivot_table(values='change',index='DATPRG',columns='CODNEG')


# In[22]:


change.head()


# In[24]:


# Função Predição
def score(self, X, y):
    y_pred = self.predict(X)
    return (y_pred==y).mean()


# In[25]:


# Função Split
def createXy(e):
    X = change[e].values
    pos = np.ceil(len(X)*0.01).astype(int)
    y = X > np.sort(X, axis=0)[::-1][pos]
    return X.reshape(-1, 1), y


# In[26]:


# Dicionário de modelos
clf = {'Tree': tree.DecisionTreeClassifier(),
       'SGD Hinge L1': SGDClassifier(loss="hinge", penalty="l1"),
       '5-NN': neighbors.KNeighborsClassifier(5),
       'SVC': svm.SVC(gamma='scale'),
       'Naive Bayes': GaussianNB(),
      'Random Forest': RandomForestClassifier()}


# In[27]:


# Dividindo dataset
scores_clf = pd.DataFrame(index=clf.keys(), columns=change.columns)
for el in change:
    for key in clf:
        X, y = createXy(el)
        this_scores = cross_val_score(clf[key], X, y, cv=5, scoring=score)
        scores_clf.loc[key, el] = this_scores.mean()


# In[30]:


# Gráfico com o score médio de cada modelo
table = pd.DataFrame(scores_clf.mean(axis=1).round(4), columns=['mean'], index=scores_clf.index).T

fig, ax = plt.subplots(1, 1, figsize=(20,6))

ax.get_xaxis().set_visible(False)
ax.set_ylim(0.98,1.001)
scores_clf.plot(table=table, ax=ax,kind='bar');

