#!/usr/bin/env python
# coding: utf-8

# **Bibliotecas**

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.tsa.seasonal import seasonal_decompose


# **Este dataset contém dados de todas as ações da bovespa do ano calendário de 1998 até 2018**
# 
# **https://www.kaggle.com/friedliver/bovespa**

# In[2]:


df = pd.read_csv('../datasets/bovespa1998-2019.csv', low_memory=False)


# **Este contém a cotação do dólar para todos os dias úteis no ano de 2018**

# In[3]:


dol = pd.read_csv('../datasets/USD_BRL Dados Históricos.csv')


# **Convertendo as colunas de data de ambos datasets para o formato certo**

# In[4]:


df['DATPRG'] = pd.to_datetime(df['DATPRG'])
dol['Data'] = pd.to_datetime(dol['Data'])
dol.set_index('Data',inplace=True)


# **Definindo o período a ser observado:**

# In[5]:


start_date = '2017-12-31'
end_date = '2018-12-31'
mask = (df['DATPRG'] > start_date) & (df['DATPRG'] <= end_date)


# In[6]:


df2 = df.loc[mask].copy()


# O período escolhido foi o ano de 2018

# **Reduzindo o dataset para somente algumas ações**

# In[7]:


sec = ['PETR4','VALE3','ITUB4','BBDC4']
df3 = df2[df2['CODNEG'].isin(sec)]


# **Tabela com as informações necessárias**

# In[8]:


# Selecionando a data, ativo e preço de fechamento
table = df3.pivot_table(index='DATPRG',columns='CODNEG',values='PREULT')

# Adicionando a cotação do dólar
table['DOL'] = dol['Último']

# Preenchendo valores nulos
table.fillna(method='pad',inplace=True)

# Convertando a cotação para o formato americano
table['DOL'] = table['DOL'].apply(lambda x: x.replace(',','.'),)
table['DOL'] = pd.to_numeric(table['DOL'])


# **Gráfico com a evolução dos preços das ações**

# In[9]:


fig, ax = plt.subplots(figsize=(20,8))
plt.plot(table, lw=3)
plt.legend(table.columns);


# **Análise Temporal**

# Petrobras

# In[12]:


seasonal_decompose(table['PETR4'],model='mul',freq=21).plot();


# Vale

# In[13]:


seasonal_decompose(table['VALE3'],model='mul',freq=21).plot();


# Itaú

# In[14]:


seasonal_decompose(table['ITUB4'],model='mul',freq=21).plot();


# Bradesco

# In[15]:


seasonal_decompose(table['BBDC4'],model='mul',freq=21).plot();


# Dólar

# In[16]:


seasonal_decompose(table['DOL'],model='mul',freq=21).plot();


# **Retorno Percentual**

# In[17]:


returns = table.pct_change()
fig, ax = plt.subplots(figsize=(20,8))
plt.plot(returns)
plt.legend(returns.columns);


# Petrobras:
# 
# Variação atípica em junho devido ao anúncio da diminução do preço do diesel.

# In[18]:


returns['PETR4'].plot()
plt.xlabel('PETR4');


# Vale

# In[20]:


returns['VALE3'].plot()
plt.xlabel('VALE3');


# Itaú:
# 
# Queda brusca em um único dia, resultado da aprovação do desdobramento de 50% das ações da empresa

# In[21]:


returns['ITUB4'].plot()
plt.xlabel('ITUB4');


# Bradesco

# In[23]:


returns['BBDC4'].plot()
plt.xlabel('BBDC4');


# Dólar

# In[24]:


returns['DOL'].plot()
plt.xlabel('DOL');


# # Razão média/desv_pad

# In[25]:


#means = pd.DataFrame(returns.tail(90).mean()) # média do último trimestre
#std = pd.DataFrame(returns.tail(90).std()) # desvio padrão do último trimestre
# média do ano
means = pd.DataFrame(returns.mean())

# desvio padrão do ano
std = pd.DataFrame(returns.std())

# razão de 'defensividade' da ação
ratios = pd.concat([means, std], axis=1).reset_index()
ratios.columns = ['Company', 'Mean', 'Std']
ratios['Ratio'] = ratios['Mean']/ratios['Std']


# In[26]:


top = ratios.sort_values('Ratio', ascending=False)


# **gráfico dos retornos vs risco**

# In[27]:


def retorno_risco(df, x, y, length=8, width=14, title=""):
    df = df.sort_values(x, ascending=False)
    plt.figure(figsize=(width,length))
    chart = sns.barplot(data=df, x=x, y=y)
    plt.title(title + "\n", fontsize=16)
    return chart

bar_plot = retorno_risco(top, 'Ratio', 'Company', title='Razão retornos vs riscos')

bar_plot;


# **Matriz de Correlação**

# In[31]:


def matriz_correl(df, dias=245):
    matriz_correl = df.tail(dias).corr()
    return matriz_correl

target_list = returns[list(ratios['Company'])]
correlation = matriz_correl(target_list)


# In[29]:


def matriz_correl(df, dias=245):
    matriz_correl = df.tail(dias).corr()
    return matriz_correl

correlation = matriz_correl(returns[list(ratios['Company'])])


# **Plot de Correlação**

# In[30]:


def plot_correl(corr, title=""):
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.subplots(figsize=(16, 8))
    
    chart = sns.heatmap(corr, mask=mask, cmap='inferno_r', center=0, linewidths=.5, annot=True, fmt='.2f',square=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Ativo')
    plt.ylabel('')
    plt.yticks(np.arange(len(corr)+1), corr)
    return chart

corr_plot = plot_correl(correlation, title='Correlação de Retornos')


# **Definindo funções**

# In[31]:


def perform_portif(pesos, media_retorno, matriz_covar): # retornos e volatilidade
    returns = np.sum(media_retorno*pesos ) *252 # 
    std = np.sqrt(np.dot(pesos.T, np.dot(matriz_covar, pesos))) * np.sqrt(252)
    return std, returns

def portfolios(num_portfolios, media_retorno, matriz_covar, taxa_livre_risco): #gerará carteiras com pesos aleatórios
    colunas = len(table.columns)
    results = np.zeros((colunas,num_portfolios))
    pesos_lista = []
    for i in range(num_portfolios):
        pesos = np.random.random(colunas)
        pesos /= np.sum(pesos)
        pesos_lista.append(pesos)
        portfolio_std_dev, portfolio_return = perform_portif(pesos,                                               media_retorno, matriz_covar)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - taxa_livre_risco) / portfolio_std_dev
    return results, pesos_lista


# **Variáveis utilizadas**

# In[32]:


returns = table.pct_change()
media_retorno = returns.mean()
matriz_covar = returns.cov()
num_portfolios = 25000
taxa_livre_risco = 0.0645 # CDI
#taxa_livre_risco = 0.0345 #Tesouro Direto - títulos pós fixados com atualização pelo IPCA - 
# cuja taxa de juros ofertada era de 3,45% aa, no início de 2018.


# **Função Criação de portfolio:**
# 
# gera uma carteira aleatória, bem como seu retorno, volatilidade e índice de Sharpe

# In[33]:


def criando_carteiras(media_retorno, matriz_covar, num_portfolios, taxa_livre_risco):
    results, pesos = portfolios(num_portfolios,media_retorno, matriz_covar, taxa_livre_risco)
    
    sharpe_max = np.argmax(results[2]) # carteira com maior sharpe
    sdp, rp = results[0,sharpe_max], results[1,sharpe_max]
    sharpe_max_pesos = pd.DataFrame(pesos[sharpe_max],index=table.columns,columns=['allocation'])
    sharpe_max_pesos.allocation = [round(i*100,2)for i in sharpe_max_pesos.allocation]
    sharpe_max_pesos = sharpe_max_pesos.T
    
    vol_min = np.argmin(results[0]) # carteira com menor vol
    sdp_min, rp_min = results[0,vol_min], results[1,vol_min]
    vol_min_pesos = pd.DataFrame(pesos[vol_min],index=table.columns,columns=['allocation'])
    vol_min_pesos.allocation = [round(i*100,2)for i in vol_min_pesos.allocation]
    vol_min_pesos = vol_min_pesos.T
    
    print ("-"*71) # linhas de separação
    print ("Distribuição da carteira pelo índice de Sharpe máximo\n")
    print ("Retorno Anualizado:", round(rp,2)) 
    print ("Volatilidade Anualizada:", round(sdp,2))
    print ("\n")
    print (sharpe_max_pesos)
    print ("-"*71)
    print ("Distribuição da carteira pela volatilidade mínima\n")
    print ("Retorno Anualizado:", round(rp_min,2))
    print ("Volatilidade Anualizada:", round(sdp_min,2))
    print ("\n")
    print (vol_min_pesos)

    plt.figure(figsize=(12, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=20, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='.',color='r',s=500, label='Índice de Sharpe Máximo')
    plt.scatter(sdp_min,rp_min,marker='.',color='g',s=500, label='Volatilidade Mínima')
    
    #plt.title('Simulacao de Portfolio Otimizado com base na Fronteira de Eficiencia')
    #plt.xlabel('volatilidade anualizada')
    #plt.ylabel('retornos anualizados')
    plt.legend(labelspacing=0.8)


# In[34]:


criando_carteiras(media_retorno, matriz_covar, num_portfolios, taxa_livre_risco)

