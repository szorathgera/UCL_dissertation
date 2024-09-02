import os

os.environ['JAX_PLATFORMS'] = 'cpu'


import sys
import time
import jax
import jax.numpy as jnp
from ac_gera import ACModel

ATFolder = '/Users/szorathgera/dissertation/AlphaTrade'

ac_model = ACModel(ATFolder, task="sell", window_index=1, action_type="pure")

# Assume the required data is loaded and processed already

total_shares = 500
starting_price = ac_model.env_params.book_data[0, 2]
volatility = 0.0125  # Example volatility
bid_ask_spread = 1741.42  # Example bid-ask spread
trading_volume = 3777546  # Example trading volume
liquidation_time = 60
num_n = 60
lambda_risk_aversion = 1e-6

eta, gamma, epsilon, single_step_variance, tau, eta_hat, kappa_hat, kappa = ac_model.calculate_ac_parameters(
total_shares, starting_price, volatility, bid_ask_spread, trading_volume, liquidation_time, num_n, lambda_risk_aversion)

trade_list = ac_model.get_trade_list(total_shares, kappa, tau, liquidation_time, num_n)

for rngInitNum in range(100, 110):
    print(f"++++ rngInitNum {rngInitNum}")
    window_index, average_price, executed_list, total_revenue = ac_model.execute_ac_strategy(trade_list, rngInitNum)
    print("Window Index:", window_index)
    print("Average Price:", average_price)
    print("Executed List:", executed_list)
    print("Total Revenue:", total_revenue)
