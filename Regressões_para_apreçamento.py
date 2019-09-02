#!/usr/bin/env python
# coding: utf-8

# In[1]:


# bibliotecas
import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# importação dos dados
df = pd.read_csv('train.csv')
df.head()


# In[3]:


# descrição dos dados
df.info()


# In[4]:


# range da idade das casas
print('Oldest House:', df['YearBuilt'].min())
print('Newest House:', df['YearBuilt'].max())


# In[5]:


# visualização de valores ausentes
fig, ax = plt.subplots(figsize=(22,6))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# In[6]:


# colunas com maior número de valores faltando
df.isna().sum().sort_values(ascending=False)


# In[7]:


# removendo colunas com muitos valores faltando
# Alley também foi removido, não por ausência de valores, mas por grande parte das casas não terem
df.drop(['PoolQC','MiscFeature','Alley','FireplaceQu','Fence'], axis=1,inplace=True)


# In[8]:


df.isna().sum().sort_values(ascending=False)


# In[9]:


# removendo mais colunas com valores NA
df.drop(['LotFrontage', 'GarageFinish', 'GarageType', 'GarageCond', 'GarageQual'], axis=1,inplace=True)


# In[10]:


df.isna().sum().sort_values(ascending=False)


# In[11]:


# removendo mais colunas
df.drop(['GarageYrBlt','BsmtFinType1','BsmtFinType2','BsmtExposure'], axis=1,inplace=True)


# In[12]:


df.isna().sum().sort_values(ascending=False)


# In[13]:


# removendo mais colunas
# removida a coluna electrical também
df.drop(['BsmtCond','BsmtQual','Electrical'], axis=1,inplace=True)


# In[14]:


# visualização de valores ausentes
fig, ax = plt.subplots(figsize=(22,6))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# In[15]:


# distribuição dos preços
fig, ax = plt.subplots(figsize=(22,6))
plt.hist(df['SalePrice'],bins=60);


# In[16]:


df['SalePrice'].skew()


# In[17]:


# "Normalizando"
logprice = np.log(df['SalePrice'])
logprice.skew()


# In[18]:


fig, ax = plt.subplots(figsize=(22,6))
plt.hist(logprice);


# In[19]:


# novo dataframe com as colunas numéricas
numcol = df.select_dtypes(include=[np.number])
print(numcol.dtypes)


# In[20]:


# análise primária de correlação das colunas numéricas
corr = numcol.corr()


# In[21]:


# As primeiras 5 colunas com maior correlação com o preço e as últimas, com menor correlação
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])


# In[22]:


plt.figure(figsize=(16,16))
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns);


# In[23]:


# Mais uma verificação de valores nulos
numcol.isna().sum().sort_values(ascending=False)


# In[24]:


# removendo a variável explicada e o Id
numcol.drop(['SalePrice','Id'],axis=1,inplace=True)


# In[25]:


# substituindo valores faltando por 0
numcol.fillna(0,inplace=True)


# In[26]:


numcol.isna().sum().sort_values(ascending=False)


# In[27]:


# regressão linear
from sklearn.model_selection import train_test_split


# In[28]:


x = numcol
y = df['SalePrice']


# In[29]:


# Divisão Treino-Teste
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


lm = LinearRegression()


# In[32]:


# Fit
lm.fit(x_train,y_train)


# In[33]:


lm.coef_


# In[34]:


predictions = lm.predict(x_test)


# In[35]:


# Scatter Plot
plt.scatter(y_test, predictions)
plt.xlabel('Y test')
plt.ylabel('Predicted y');


# In[36]:


# R^2 do teste
lm.score(x_test,y_test)


# In[37]:


# R^2 do treino
lm.score(x_train,y_train)


# In[38]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[39]:


sns.distplot(y_test-predictions, bins=20);


# In[40]:


coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff_df.sort_values(by=['Coefficient'],ascending=False).head()


# In[41]:


coeff_df.sort_values(by=['Coefficient']).head()


# In[42]:


# novo dataframe com as colunas numéricas
catcol = df.select_dtypes(include=[np.object])
print(catcol.dtypes)


# In[43]:


catcold = pd.get_dummies(catcol,drop_first=True) 


# In[44]:


catcold.head()


# In[45]:


# Visualização de PCA
from sklearn.preprocessing import StandardScaler
catcold_std = StandardScaler().fit_transform(catcold)


# In[46]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(catcold_std)


# In[47]:


Y_sklearn


# In[80]:


# Scatter plot do PCA
plt.figure(figsize=(8,6))
plt.scatter(Y_sklearn[:,0],Y_sklearn[:,1],
           # color
           c=df['SalePrice'],
           # colormap
           cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.colorbar();


# In[49]:


# Regressão Logística


# In[50]:


# Juntando os dataframes, numérico e categórico (agora com dummies)
newdf = pd.concat([numcol,catcold],axis=1)


# In[51]:


newdf.head()


# In[71]:


# Separando os preços por faixas
faixas = pd.cut(df['SalePrice'],3,labels=['1st','2nd','3rd'])
faixas


# In[72]:


# Divisão Treino-Teste
X_train,X_test,Y_train,Y_test = train_test_split(newdf,faixas,test_size=0.30,random_state=101)


# In[73]:


# Treino
from sklearn.linear_model import LogisticRegression


# In[74]:


# Regressão logística multinomial, pois é um problema de multiclasse
logmodel = LogisticRegression(multi_class='multinomial', solver='newton-cg')
logmodel.fit(X_train,Y_train)


# In[75]:


predictions = logmodel.predict(X_test)


# In[76]:


from sklearn.metrics import classification_report


# In[77]:


# Resultado
print(classification_report(Y_test,predictions))


# In[ ]:





# In[ ]:


# Lendo a descrição das colunas, descobri que tem casas listadas com amianto (asbestos)
df.loc[(df['Exterior1st']=='AsbShng')|(df['Exterior2nd']=='AsbShng')]


# In[ ]:


df['Exterior1st'].value_counts()


# In[ ]:


df['Exterior2nd'].value_counts()


# In[ ]:


# Dummies para para exterior
exterior = pd.get_dummies(df['Exterior1st'],drop_first=True) 


# In[ ]:


exterior.head()

