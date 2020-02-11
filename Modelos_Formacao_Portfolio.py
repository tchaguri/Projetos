#!/usr/bin/env python
# coding: utf-8

# **Bibliotecas**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# **Datasets**

# In[2]:


df1 = pd.read_csv('BBDC4.csv')
df2 = pd.read_csv('ITUB4.SA.csv')
df3 = pd.read_csv('PETR4.SA.csv')
df4 = pd.read_csv('VALE3.SA.csv')
df5 = pd.read_csv('PETR3.SA.csv')
dol = pd.read_csv('USD_BRL Dados Históricos.csv')


# **Ativos**

# In[3]:


df1.drop(axis=1, columns=['Open','High','Low','Close','Volume'], inplace=True)
df1.rename(columns={'Adj Close': 'BBDC4','Date': 'Data'},inplace=True)
df1['ITUB4'] = df2['Adj Close']
df1['PETR4'] = df3['Adj Close']
df1['VALE3'] = df4['Adj Close']
df1['PETR3'] = df5['Adj Close']
df1['Data'] = pd.to_datetime(df1['Data'])
df1.set_index('Data',inplace=True)


# **Dolar**

# In[4]:


dol['Dollar'] = dol['Último'].apply(lambda x: x.replace(',','.'),)
dol['Data'] = pd.to_datetime(dol['Data'])
dol['Dollar'] = pd.to_numeric(dol['Dollar'])
dol.set_index('Data',inplace=True)
dol.sort_index(inplace=True)


# **Tabela com as informações necessárias**

# In[5]:


table = df1.pivot_table(index='Data',values=['ITUB4','BBDC4','PETR4','VALE3','PETR3',])


# **Gráfico com a evolução dos preços das ações**

# In[6]:


fig, ax = plt.subplots(figsize=(20,8))
plt.plot(table, lw=3)
plt.legend(table.columns);


# In[7]:


table['DOL'] = dol['Dollar']
table.fillna(method='pad', axis=0, inplace=True)


# # Calculo dos retornos
# **equação 2.1**

# In[8]:


returns = table.pct_change()
fig, ax = plt.subplots(figsize=(20,8))
plt.plot(returns)
plt.legend(returns.columns);


# # Razão média/desvio padrão

# In[9]:


means = pd.DataFrame(returns.mean())
std = pd.DataFrame(returns.std())
ratios = pd.concat([means, std], axis=1).reset_index()
ratios.columns = ['Company', 'Mean', 'Std']
ratios['Ratio'] = ratios['Mean']/ratios['Std'] # razão de 'defensividade' da ação


# In[10]:


top = ratios.sort_values('Ratio', ascending=False)


# **gráfico dos retornos vs risco**

# In[11]:


def barchart(df, x, y, length=8, width=14, title=""):
    df = df.sort_values(x, ascending=False)
    plt.figure(figsize=(width,length))
    chart = sns.barplot(data=df, x=x, y=y)
    plt.title(title + "\n", fontsize=16)
    return chart

bar_plot = barchart(top, 'Ratio', 'Company', title='Razão Retornos vs Riscos')

bar_plot;


# **Matriz de Correlação**

# In[12]:


def corr_matrix(df, days=30): #estabelecer período de observação (days) - padrão 30
    corr_matrix = df.tail(days).corr() #matriz de CORRELAÇÃO
    return corr_matrix

target_list = returns[list(ratios['Company'])]
correlation = corr_matrix(target_list)


# **Plot de Correlação**

# In[13]:


def correlation_plot(corr, title=""):
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.subplots(figsize=(20, 20))
    
    chart = sns.heatmap(corr, mask=mask, cmap='inferno_r', center=0, linewidths=.5, annot=True, fmt='.2f',square=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Ativos')
    plt.ylabel('')
    plt.yticks(np.arange(len(corr)+1), corr)
    return chart

corr_plot = correlation_plot(correlation, title='Correlação de Retornos')


# # Markowitz

# **Retorno e volatilidade.**

# In[14]:


def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252 # ano dias úteis
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # ano dias úteis
    return std, returns


# **Gerando carteiras com pesos aleatórios.**

# In[15]:


def random_port_generator(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    dim_res = len(table.columns) # quantidade de títulos
    results = np.zeros((dim_res,num_portfolios)) # matriz de zeros número de colunas X numero de portfólios (que é dado)
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(dim_res) # peso w de [0, 1[ para o número de colunas
        weights /= np.sum(weights) # divide os pesos pela soma
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_performance(weights,                                               mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


# In[16]:


mean_returns = returns.mean()
cov_matrix = returns.cov() # matriz de COVARIÂNCIA
num_portfolios = 500000 # quantidade de carteiras geradas
risk_free_rate = 0.0642 # CDI


# In[17]:


#A função definida a seguir, em primeiro lugar, gera um porfólio (carteira)
# aleatório e obtém os resultados (retornos, volatilidade e índice de 
# Sharpe do portfólio) e os pesos usados para a obtenção dos resultados correspondentes.
def efficient_frontier_simulation(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_port_generator(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2]) # carteira com maior sharpe
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0]) # carteira com menor vol
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("#"*60) # linhas de separação
    print("Alocacao da Carteira pelo Indice de Sharpe Maximo\n")
    print("Retorno Anualizado:", round(rp,2)) 
    print("Volatilidade Anualizada:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("#"*60)
    print("Alocacao da Carteira pela Volatilidade Minima\n")
    print("Retorno Anualizado:", round(rp_min,2))
    print("Volatilidade Anualizada:", round(sdp_min,2),)
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(12, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=20, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='.',color='r',s=500, label='Indice de Sharpe Maximo')
    plt.scatter(sdp_min,rp_min,marker='.',color='g',s=500, label='Volatilidade Minima')
    
    plt.xlabel('Risco')
    plt.ylabel('Retorno')
    plt.legend(labelspacing=0.8)


# In[18]:


efficient_frontier_simulation(mean_returns, cov_matrix, num_portfolios, risk_free_rate)


# # Paridade de Risco

# **Bibliotecas**

# In[19]:


from __future__ import division
from numpy.linalg import inv,pinv
from scipy.optimize import minimize


# **Cálculo de contribuição de risco**

# In[20]:


def portfolio_var(w,E):
    w = np.matrix(w) # calculo do risco do portfolio
    return (w*E*w.T)[0,0]


# In[21]:


def risk_contribuition(w,E):
    w = np.matrix(w) # calculo da contribuição individual para o risco total
    sigma = np.sqrt(portfolio_var(w,E))
    # Marginal Risk Contribuition
    MRC = E*w.T
    # Risk Contribuition
    RC = np.multiply(MRC,w.T)/sigma
    return RC


# **Paridade de risco**

# In[22]:


def risk_par(x,pars):
    # calculate portfolio risk
    E = pars[0] # covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p = np.sqrt(portfolio_var(x,E)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = risk_contribuition(x,E)
    J = sum(np.square(asset_RC - risk_target.T))[0,0]*num_portfolios # sum of squared error
    return J


# **Restrições**

# In[23]:


def constraint_weight(x):
    return np.sum(x)-1.0

def constraint_long(x):
    return x


# In[24]:


w0 = np.ones(6)*1.0/6
x_t = [1/6] * 6


# **matriz de covariância**

# In[25]:


E = np.matrix(cov_matrix.get_values())


# In[27]:


cons = ({'type': 'eq', 'fun': constraint_weight},
{'type': 'ineq', 'fun': constraint_long})
objective_function= minimize(risk_par, w0, args=[E,x_t], method='SLSQP',constraints=cons, options={'disp': True})
w_risk_par = np.asmatrix(objective_function.x)


# In[31]:


rp_alloc = pd.DataFrame(data=w_risk_par,columns=table.columns)


# **resultado**

# In[34]:


print("Alocação segundo Paridade de Risco:\n")
print(rp_alloc)

