# baseline_models.py

import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from gymnax_exchange.jaxen.exec_env import ExecutionEnv

class ACModel:
    def __init__(self, ATFolder, task="sell", window_index=1, action_type="pure"):
        self.env = ExecutionEnv(ATFolder, task, window_index, action_type)
        self.env_params = self.env.default_params

    def calculate_ac_parameters(self, total_shares, starting_price, volatility, bid_ask_spread, trading_volume, liquidation_time, num_n, lambda_risk_aversion):
        trad_days = 252
        daily_volat = volatility / np.sqrt(trad_days)
        single_step_variance = (daily_volat * starting_price) ** 2
        epsilon = bid_ask_spread / 2
        eta = bid_ask_spread / (0.01 * trading_volume)
        gamma = bid_ask_spread / (0.1 * trading_volume)
        tau = liquidation_time / num_n
        eta_hat = eta - (0.5 * gamma * tau)
        kappa_hat = np.sqrt((lambda_risk_aversion * single_step_variance) / eta_hat)
        kappa = np.minimum(np.arccosh((((kappa_hat ** 2) * (tau ** 2)) / 2) + 1) / tau, 10)
        return eta, gamma, epsilon, single_step_variance, tau, eta_hat, kappa_hat, kappa

    def get_trade_list(self, total_shares, kappa, tau, liquidation_time, num_n):
        trade_list = np.zeros(num_n)
        ftn = 2 * np.sinh(0.5 * kappa * tau)
        ftd = np.sinh(kappa * liquidation_time)
        if np.isinf(ftd):
            ftd = np.sinh(np.minimum(kappa * liquidation_time, 10))
        ft = (ftn / ftd) * total_shares
        for i in range(1, num_n + 1):
            st = np.cosh(kappa * (liquidation_time - (i - 0.5) * tau))
            if np.isinf(st):
                st = np.cosh(np.minimum(kappa * (liquidation_time - (i - 0.5) * tau), 10))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list

    def execute_ac_strategy(self, trade_list, rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        obs, state = self.env.reset(key_reset, self.env_params)
        executed_list = []

        for i in range(len(trade_list)):
            shares_to_sell = trade_list[i]

            def ac_action(state, shares_to_sell, key_policy):
                remainingTime = self.env_params.episode_time - jnp.array((state.time - state.init_time)[0], dtype=jnp.int32)
                marketOrderTime = jnp.array(60, dtype=jnp.int32)
                ifMarketOrder = (remainingTime <= marketOrderTime)

                stepQuant = jnp.ceil(shares_to_sell).astype(jnp.int32)
                limit_quants = jax.random.permutation(key_policy, jnp.array([stepQuant - stepQuant // 2, stepQuant // 2, stepQuant // 4, stepQuant // 4]), independent=True)
                market_quants = jnp.array([stepQuant, stepQuant, stepQuant, stepQuant])

                quants = jnp.where(ifMarketOrder, market_quants, limit_quants)
                return quants

            ac_action_value = ac_action(state, shares_to_sell, key_policy)

            obs, state, reward, done, info = self.env.step(key_step, state, ac_action_value, self.env_params)
            executed_list.append(info["quant_executed"])

            if done:
                break

        return info['window_index'],info['average_price'], info['slippage_rm'], info['price_adv_rm'], info['price_drift_rm'], info['advantage_reward'],info['total_revenue'],info['task_to_execute'] ,executed_list
