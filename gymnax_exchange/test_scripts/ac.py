import os 



import sys
sys.path.append('/Users/szorathgera/Dissertation/AlphaTrade/')

import numpy as np
import pandas as pd
import random
import collections
import jax
import jax.numpy as jnp
from gymnax_exchange.jaxen.exec_env import *

env=ExecutionEnv(ATFolder,"sell")

message_data_path = "../../data/AMZN_2012-06-21_34200000_57600000_message_10.csv"
orderbook_data_path = "../../data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

message_data = pd.read_csv(message_data_path)
orderbook_data = pd.read_csv(orderbook_data_path)

# Display the first few rows of the data
message_data.head(), orderbook_data.head()


# ------------------------------------------------ Financial Parameters --------------------------------------------------- #

ANNUAL_VOLAT = 0.12                                # Annual volatility in stock price
BID_ASK_SP = 1 / 8                                 # Bid-ask spread
DAILY_TRADE_VOL = 5e6                              # Daily trading volume  
TRAD_DAYS = 250                                    # Number of trading days in a year
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)    # Daily volatility in stock price


# ----------------------------- Parameters for the Almgren and Chriss Optimal Execution Model ----------------------------- #

TOTAL_SHARES = 1000000                                               # Total number of shares to sell
STARTING_PRICE = 50                                                  # Starting price per share
LLAMBDA = 1e-6                                                       # Trader's risk aversion
LIQUIDATION_TIME = 60                                                # How many days to sell all the shares. 
NUM_N = 60                                                           # Number of trades
EPSILON = BID_ASK_SP / 2                                             # Fixed Cost of Selling.
SINGLE_STEP_VARIANCE = (DAILY_VOLAT  * STARTING_PRICE) ** 2          # Calculate single step variance
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)                          # Price Impact for Each 1% of Daily Volume Traded
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)                         # Permanent Impact Constant

# ----------------------------------------------------------------------------------------------------------------------- #


class AC():
    
    def __init__(self, randomSeed = 0,
                 lqd_time = LIQUIDATION_TIME,
                 num_tr = NUM_N,
                 lambd = LLAMBDA):
        
        # Set the random seed
        random.seed(randomSeed)
        
        # Initialize the financial parameters so we can access them later
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT
        
        # Initialize the Almgren-Chriss parameters so we can access them later
        self.total_shares = TOTAL_SHARES
        self.startingPrice = STARTING_PRICE
        self.llambda = lambd
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA
        
        # Calculate some Almgren-Chriss parameters
        self.tau = self.liquidation_time / self.num_n 
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
        self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau

        # Set the variables for the initial state
        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n
        self.logReturns = collections.deque(np.zeros(6))
        
        # Set the initial impacted price to the starting price
        self.prevImpactedPrice = self.startingPrice

        # Set the initial transaction state to False
        self.transacting = False
        
        # Set a variable to keep trak of the trade number
        self.k = 0
        
#         self.lagCoeffs = np.array([1.5, -0.9, 0.8, -0.6, 0.5, -0.3])     
#         self.alphas = np.array([1, 0.4, 0.3, 0.2, 0.1, 0.1]) * 30
#         self.a_deque = collections.deque(np.zeros(len(self.alphas)))
#         self.returns_deque = collections.deque(np.zeros(len(self.lagCoeffs)))                
#         self.unaffected_returns_deque = collections.deque(np.zeros(len(self.lagCoeffs)))   
#         self.logReturns = collections.deque(np.zeros(len(self.lagCoeffs)))
        
        # Constant multiplier for the action returned by the Actor-Critic Model
#         self.constantSharesToSell = (self.total_shares / self.num_n) * 12
        
    def get_ac_average_price(self, rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    
        # Reset the environment and get initial observations and state
        obs, state = env.reset(key_reset, env_params)
        
        # List to store execution prices
        executed_list = []
        executed_prices = []
        def ac_strategy(self, state, env_params, step):
            # Calculate some Almgren-Chriss parameters
            self.tau = self.liquidation_time / self.num_n 
            self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
            self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
            self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau

            trade_list = jnp.zeros(self.num_n)
            ftn = 2 * jnp.sinh(0.5 * self.kappa * self.tau)
            ftd = jnp.sinh(self.kappa * self.liquidation_time)
            ft = (ftn / ftd) * self.total_shares

            for i in range(1, self.num_n + 1):       
                st = jnp.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
                trade_list[i - 1] = st
            trade_list = trade_list * ft
            step_quant = trade_list[step]
            return step_quant

        for i in range(self.num_n):
            print("---" * 20)
            print("window_index ", state.window_index)
            
            # Calculate the quantity to trade using the AC strategy
            quants = ac_strategy(state, env_params, state.step_counter)
            
            current_price = state.best_asks[state.step_counter] if quants > 0 else state.best_bids[state.step_counter]
            execution_price = self.apply_temporary_impact(current_price, quants)

            state.best_asks[state.step_counter] = self.apply_permanent_impact(state.best_asks[state.step_counter], quants)
            state.best_bids[state.step_counter] = self.apply_permanent_impact(state.best_bids[state.step_counter], quants)

            start = time.time()
            obs, state, reward, done, info = env.step(key_step, state, jnp.array([quants]), env_params)
            print(f"Time for {i} step: \n", time.time() - start)
            print("executed ", info["quant_executed"])
            
            executed_list.append(info["quant_executed"])
            executed_prices.append(execution_price)

            
            # Check if the episode has ended
            if done:
                break
        
        return info['window_index'], info['average_price'], executed_list

        

    def reset(self, seed = 0, liquid_time = LIQUIDATION_TIME, num_trades = NUM_N, lamb = LLAMBDA):
        
        # Initialize the environment with the given parameters
        self.__init__(randomSeed = seed, lqd_time = liquid_time, num_tr = num_trades, lambd = lamb)
        
        # Set the initial state to [0,0,0,0,0,0,1,1]
        self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, \
                                                               self.shares_remaining / self.total_shares])
        return self.initial_state

    

    def permanentImpact(self, sharesToSell):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        pi = self.gamma * sharesToSell
        return pi

    
    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        ti = (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)
        return ti
    
    def get_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall according to equation (8) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell
        tt = (self.eta_hat / self.tau) * self.totalSSSQ
        return ft + st + tt

    
    def get_AC_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell        
        tt = self.eta_hat * (sharesToSell ** 2)       
        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) \
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))       
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)   
        fot = nft / dft       
        return ft + st + (tt * fot)  
        
    
    def get_AC_variance(self, sharesToSell):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * (self.singleStepVariance) * (sharesToSell ** 2)                        
        nst  = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(self.kappa * (self.liquidation_time - self.tau)) \
               - self.liquidation_time * np.sinh(self.kappa * self.tau)        
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)        
        st = nst / dst
        return ft * st
        
        
    def compute_AC_utility(self, sharesToSell):    
        # Calculate the AC Utility according to pg. 13 of the AC paper
        if self.liquidation_time == 0:
            return 0        
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V
    
    
    def get_trade_list(self):
        # Calculate the trade list for the optimal strategy according to equation (18) of the AC paper
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares
        for i in range(1, self.num_n + 1):       
            st = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list
     
        
    def observation_space_dimension(self):
        # Return the dimension of the state
        return 8
    
    
    def action_space_dimension(self):
        # Return the dimension of the action
        return 1
    
    
    def stop_transactions(self):
        # Stop transacting
        self.transacting = False            