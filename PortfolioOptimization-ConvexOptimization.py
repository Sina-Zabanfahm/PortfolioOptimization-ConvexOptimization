#!/usr/bin/env python
# coding: utf-8

# In[238]:


import cvxpy as cp
import pandas as pd 
import numpy as np
import seaborn as sns
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

import warnings
warnings.simplefilter('ignore')

import json


from statsmodels import regression
from sklearn.linear_model import LinearRegression


# In[239]:


import yfinance as yf
start_date = "2021-03-01"
end_date = "2023-03-01"
with open('portfolio_optimize.json') as f:
  cols = json.load(f)

combined = cols['shorts'] +cols['longs']
short_cols = cols['shorts']
long_cols = cols['longs']


# In[240]:


data  = yf.download(combined, start_date, end_date)['Close']
data.set_index(pd.to_datetime(data.index),inplace = True)


# In[241]:


data.dropna(inplace = True,axis =1)


# In[242]:


data.shape


# In[243]:


short_pos = []
long_pos = []
for c in data.columns:
    if c in short_cols:
        short_pos.append(c)
    if c in long_cols:
        long_pos.append(c)


# 

# # New Section

# In[247]:


from pandas._libs.lib import is_float_array
class Portfolio:


    def __init__ (self, hist_data,short_pos, long_pos, budget=100000000,
                  risk_free_rate=0.0425, all_invested = True,non_neg =False,
                  start_date ="2022-03-01" ,end_date ="2023-03-01"):
      

        """
        Initializing the data: 
            hist_data is historical data of securities
            short_pos indicates stocks that will be shorted
            long_pos indicates stocks that are in long positions
            buddget indicates the amount that will be invested
            all_invested checks if we are investing all the capital.

        """
        
        
        self.hist_data = hist_data
        self.short_pos = short_pos
        self.long_pos = long_pos
        self.budget = budget
        self.risk_free_rate = risk_free_rate
        self.optimal_weights = pd.DataFrame()
        self.all_invested = all_invested
        self.start_date = start_date
        self.end_date = end_date

        self.non_neg = non_neg

    def optimize(self):


        returns = self.hist_data.pct_change().fillna(0)
        Cov = self.hist_data.pct_change().cov().values
        num_assets = len(self.hist_data.columns)
        y = cp.Variable(num_assets, nonneg=self.non_neg)
        ret_mean = returns.mean()

        # Define the objective function - maximize the quadratic form

        r = self.risk_free_rate
        k = cp.sum(y)
        e = np.array(np.repeat(1,len(self.hist_data.columns)))
        """
          We cannot use the usual form of Sharpe ratio does not work for cvxpy.
          We use an equivalten formulation of the Sharpe ratio.
          See :http://yetanothermathprogrammingconsultant.blogspot.com/2016/08/portfolio-optimization-maximize-sharpe.html
          or, see: Optimization Methods in Finance (lecture notes by G. Cornuejols, et al.) section 8.2
          
          Equivalence requires some conditions on risk_free_rate and space of feasabile portfolios.
        """


        sharpe = cp.quad_form(y, cp.psd_wrap(Cov))


        objective = cp.Maximize(-sharpe)


        # Define the constraints
        prices = data.iloc[-1]
        
        negPrices = [0]*len(self.hist_data.columns)
        posPrices = [0]*len(self.hist_data.columns)

        for i in range(len(self.hist_data.columns)):
            if self.hist_data.columns[i] in self.short_pos:
                negPrices[i] = self.hist_data[self.hist_data.columns[i]][-1]
            if self.hist_data.columns[i] in self.long_pos:
                posPrices[i] = self.hist_data[self.hist_data.columns[i]][-1]
        negPrices = np.array(negPrices)
        posPrices = np.array(posPrices)

        #print(negPrices)
        
        weights = y

        if (self.all_invested == True):
                constraints = [cp.sum(y@prices) == self.budget*cp.sum(y),

                    cp.sum(y)>=0.000000001,
                    cp.sum(y @ (ret_mean-r*e))==1,
                ]
        else:
                constraints = [cp.sum(y@prices) == self.budget*cp.sum(y),
                    cp.sum(y@negPrices) == self.budget*cp.sum(y)*-0.3,
                    cp.sum(y@posPrices) == self.budget*cp.sum(y)*1.3,
                    cp.sum(y)>=0.000000001,
                    cp.sum(y @ (ret_mean-r*e))==1,
                ]  
         
        # Define the problem and solve it
        problem = cp.Problem(objective, constraints)

        """
          This is a conic optization problem, SCS solver will do. Alternatively ECOS solver 
          works too.
        """
        try:
              problem.solve(solver=cp.SCS)
        except:
              print("Exception occured in solving the optimization problem.")
              print("Experiment with different solver and check the constrains")
        # Retrieve the optimized weights

        
        self.optimal_weights = y.value
        self.optimal_weights = self.optimal_weights/np.sum(self.optimal_weights)
        #print("Optimized Portfolio Weights:")
        

    def get_weights(self):
        return self.optimal_weights;
    
    def get_data(self):
        return self.hist_data
    
    def get_risk_free_rate(self):
        return self.risk_free_rate

    def print_weights(self):

        for i, ticker in enumerate(self.hist_data.columns):
            print(f"{ticker}: {self.optimal_weights[i]}")
    
    def get_dates(self):

        #Returns the start and end_date of the training period

        return self.start_date,self.end_date
    
    def get_budget(self):
        return np.sum((self.hist_data.iloc[-1]@self.optimal_weights))
        
    


# In[248]:


portfolio = Portfolio(data,short_pos,long_pos,start_date = start_date, end_date = end_date)
portfolio.optimize()
portfolio.print_weights()


# In[249]:


portfolio.get_budget()


# In[250]:


class Metrics:


    def __init__ (self,portfolio):
        self.portfolio = portfolio
        self.sharpe = None


    def compute_sharpe(self):
        weights = self.portfolio.get_weights()
        r = self.portfolio.get_risk_free_rate()
        data = self.portfolio.get_data()   
        #daily returns
        returns = data.pct_change().fillna(0)
        #covariance of returns
        cov = returns.cov()
        mean_ret = returns.mean().values

        P_ret = np.sum(mean_ret * weights)
        P_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        self.sharpe = (P_ret - r) / P_vol
   

    def get_sharpe(self):
        
        try:
            return self.sharpe
        
        except:
            self.compute_sharpe()
            return self.sharpe

    def print_sharpe(self):

        print(f"Sharpe ratio of the portfolio is {self.get_sharpe():.2f}.")

    def cummulative_return (self):
        data = self.portfolio.get_data()
        data.set_index(pd.to_datetime(data.index), inplace = True)
        cols = data.columns
        data = yf.download(list(cols), start=start_date, end=end_date)['Close']

    def compare_values(self,benchmark_ticker = 'SPY',start_date = "", end_date = ""):

        # Define the start date and end date (today's date)
        start_date = self.portfolio.get_dates()[1]  # Replace yyyy, mm, dd with your desired start date
        end_date = datetime.datetime.today()

        # Fetch SPY data
        benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date)['Close']

        data = self.portfolio.get_data()
        #data.set_index(pd.to_datetime(data.index), inplace = True)
        cols = data.columns
        data = yf.download(list(cols), start=start_date, end=end_date)['Close']
        weights = portfolio.get_weights()
        d=(data*weights).sum(axis = 1)
        budget = portfolio.get_budget()
        plt.plot(d)
        
        
        plt.plot(d, label='Portfolio')
        plt.plot(benchmark * d.iloc[0]/benchmark.iloc[0], label=benchmark_ticker)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title(f'Porfolio vs. {benchmark_ticker}')
        plt.legend()
        plt.show()
    
    
        



        






# In[251]:


metrics = Metrics(portfolio)
metrics.compute_sharpe()
metrics.print_sharpe()
metrics.compare_values()

