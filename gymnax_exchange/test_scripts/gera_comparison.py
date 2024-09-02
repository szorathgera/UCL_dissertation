import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import sys
from re import L
import time 
sys.path.append('/Users/szorathgera/dissertation/AlphaTrade')
sys.path.append('/Users/szorathgera/dissertation/')
sys.path.append('/Users/szorathgera/dissertation/purejaxrl')

import chex
import faulthandler; faulthandler.enable()
import platform
if platform.system() == 'Darwin' and 'arm' in platform.processor():
    print("Running on Apple Silicon, skipping GPU assertion")
else:
    chex.assert_gpu_available(backend=None)

#chex.assert_gpu_available(backend=None)
from jax import config # Code snippet to disable all jitting.
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)
from gymnax_exchange.jaxen.test_exec_env import *
import json
# ============== testing scripts ===============

paramsFile_ppo = '/Users/szorathgera/dissertation/AlphaTrade/gymnax_exchange/jaxrl/s5_ppo_train/params_file_ethereal-cherry-165_08-21_01-21'
paramsFile_lstm = '/Users/szorathgera/dissertation/AlphaTrade/gymnax_exchange/jaxrl/lstm_ppo_train/params_file_stoic-puddle-167_08-21_20-00'
paramsFile_gru = '/Users/szorathgera/dissertation/AlphaTrade/gymnax_exchange/jaxrl/gru_ppo_train/params_file_quiet-glade-170_08-22_16-43'
# paramsFile = '/homes/80/kang/AlphaTrade/params_file_dutiful-thunder-5_07-21_18-48'

# def twapV3(state, env_params):
#     # ---------- ifMarketOrder ----------
#     remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
#     marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
#     ifMarketOrder = (remainingTime <= marketOrderTime)
#     # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
#     # ---------- ifMarketOrder ----------
#     # ---------- quants ----------
#     remainedQuant = state.task_to_execute - state.quant_executed
#     remainedStep = state.max_steps_in_episode - state.step_counter
#     stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
#     limit_quants = jax.random.permutation(key_policy, jnp.array([stepQuant//2,stepQuant-stepQuant//2,stepQuant//2,stepQuant-stepQuant//2]), independent=True)
#     market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
#     quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
#     # ---------- quants ----------
#     return jnp.array(quants) 

from ac_gera import ACModel


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing_small'
        ATFolder = '/Users/szorathgera/dissertation/AlphaTrade'

    outputfile = 'index_2_loop_test_newww.txt'

    

    ac_model = ACModel(ATFolder, task="sell", window_index=2, action_type="pure")


    total_shares = 500
    starting_price = ac_model.env_params.book_data[0, 2]
    volatility = 0.0125  # Example volatility
    bid_ask_spread = 1741.42  # Example bid-ask spread
    trading_volume = 3777546  # Example trading volume
    liquidation_time = 30
    num_n = 60
    lambda_risk_aversion = 1e-6

    eta, gamma, epsilon, single_step_variance, tau, eta_hat, kappa_hat, kappa = ac_model.calculate_ac_parameters(
    total_shares, starting_price, volatility, bid_ask_spread, trading_volume, liquidation_time, num_n, lambda_risk_aversion)

    trade_list = ac_model.get_trade_list(total_shares, kappa, tau, liquidation_time, num_n)

        
    env=ExecutionEnv(ATFolder,"sell", 2, "pure")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)
    assert env.task_size == 500
    import time; timestamp = str(int(time.time()))

    # PPO average price
    
    def get_ppo_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        obs,state=env.reset(key_reset,env_params)
        ppo_config = {
                "LR": 2.5e-5,
                "NUM_ENVS": 1, # CAUTION !!!
                "NUM_STEPS": 1,
                "TOTAL_TIMESTEPS": 1e7,
                "UPDATE_EPOCHS": 1,
                "NUM_MINIBATCHES": 1,
                "GAMMA": 0.99,
                "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2,
                "ENT_COEF": 0.1,
                "VF_COEF": 0.5,
                "MAX_GRAD_NORM": 0.5,
                "ANNEAL_LR": True,
                "DEBUG": True,
                
                "ENV_NAME": "alphatradeExec-v0",
                "NORMALIZE_ENV": False,
                "ENV_LENGTH": "oneWindow",
                "ATFOLDER": ATFolder,
                "TASKSIDE":'sell',
                "LAMBDA":0.0,
                "TASK_SIZE":500,
            }
        import flax
        from gymnax_exchange.jaxrl.ppoS5ExecCont import ActorCriticS5
        from gymnax_exchange.jaxrl.ppoS5ExecCont import StackedEncoderModel, ssm_size, n_layers
        network = ActorCriticS5(env.action_space(env_params).shape[0], config=ppo_config)
        init_hstate = StackedEncoderModel.initialize_carry(ppo_config["NUM_ENVS"], ssm_size, n_layers)
            
        # ===================================================
        # CHOICE ONE
        with open(paramsFile_ppo, 'rb') as f:
            restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"pramas restored")
        # ---------------------------------------------------
        # init_x = (
        #     jnp.zeros(
        #         (1, ppo_config["NUM_ENVS"], *env.observation_space(env_params).shape)
        #     ),
        #     jnp.zeros((1, ppo_config["NUM_ENVS"])),
        # )
        # network_params = network.init(key_policy, init_hstate, init_x)
        # restored_params = network_params
        # CHOICE OTWO
        # ===================================================
        
        
        init_done = jnp.array([False]*ppo_config["NUM_ENVS"])
        ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
        assert len(ac_in[0].shape) == 3
        hstate, pi, value = network.apply(restored_params, init_hstate, ac_in)
        print("Network Carry Initialized")
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None) # CAUTION about the [0,0,:], only works for num_env=1
        print(f"-------------\nPPO 0th actions are: {action} with sum {action.sum()}")
        obs,state,reward,done,info=env.step(key_step, state, action, env_params)
        print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
        excuted_list = []
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from trained network ----------
            ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
            assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
            assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
            hstate, pi, value = network.apply(restored_params, hstate, ac_in) 
            action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
            # ---------- acion from trained network ----------
            # ==================== ACTION ====================    
            print(f"-------------\nPPO {i}th actions are: {action} with sum {action.sum()}")
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'],info['average_price'], info['slippage_rm'], info['price_adv_rm'], info['price_drift_rm'], info['advantage_reward'],info['total_revenue'],info['task_to_execute'] ,excuted_list
    

    def get_ppo_lstm_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        obs,state=env.reset(key_reset,env_params)
        lstm_config = {
                "LR": 1.017e-05,
                "NUM_ENVS": 1, # CAUTION !!!
                "NUM_STEPS": 1,
                "TOTAL_TIMESTEPS": 1e7,
                "UPDATE_EPOCHS": 1,
                "NUM_MINIBATCHES": 1,
                "GAMMA": 0.974,
                "GAE_LAMBDA": 0.95,
                "CLIP_EPS":  0.218,
                "ENT_COEF": 0.00387,
                "VF_COEF": 0.6812,
                "MAX_GRAD_NORM": 0.5,
                "ANNEAL_LR": True,
                "DEBUG": True,
                
                "ENV_NAME": "alphatradeExec-v0",
                "NORMALIZE_ENV": False,
                "ENV_LENGTH": "oneWindow",
                "ATFOLDER": ATFolder,
                "TASKSIDE":'sell',
                "LAMBDA":0.0,
                "TASK_SIZE":500,
            }
        import flax
        from gymnax_exchange.jaxrl.lstm_ppo_gera import ActorCriticRNN
        from gymnax_exchange.jaxrl.lstm_ppo_gera import ScannedRNN
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=lstm_config)
        init_hstate = ScannedRNN.initialize_carry(lstm_config["NUM_ENVS"], 128)
            
        # ===================================================
        # CHOICE ONE
        with open(paramsFile_lstm, 'rb') as f:
            restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"pramas restored")
        # ---------------------------------------------------
        # init_x = (
        #     jnp.zeros(
        #         (1, ppo_config["NUM_ENVS"], *env.observation_space(env_params).shape)
        #     ),
        #     jnp.zeros((1, ppo_config["NUM_ENVS"])),
        # )
        # network_params = network.init(key_policy, init_hstate, init_x)
        # restored_params = network_params
        # CHOICE OTWO
        # ===================================================
        
        
        init_done = jnp.array([False]*lstm_config["NUM_ENVS"])
        ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
        assert len(ac_in[0].shape) == 3
        hstate, pi, value = network.apply(restored_params, init_hstate, ac_in)
        print("Network Carry Initialized")
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None) # CAUTION about the [0,0,:], only works for num_env=1
        print(f"-------------\nLSTM 0th actions are: {action} with sum {action.sum()}")
        obs,state,reward,done,info=env.step(key_step, state, action, env_params)
        print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
        excuted_list = []
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from trained network ----------
            ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
            assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
            assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
            hstate, pi, value = network.apply(restored_params, hstate, ac_in) 
            action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
            # ---------- acion from trained network ----------
            # ==================== ACTION ====================    
            print(f"-------------\nLSTM {i}th actions are: {action} with sum {action.sum()}")
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'],info['average_price'], info['slippage_rm'], info['price_adv_rm'], info['price_drift_rm'], info['advantage_reward'],info['total_revenue'],info['task_to_execute'] ,excuted_list
    
    def get_ppo_gru_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        obs,state=env.reset(key_reset,env_params)
        gru_config = {
                "LR": 1.116e-5,
                "NUM_ENVS": 1, # CAUTION !!!
                "NUM_STEPS": 1,
                "TOTAL_TIMESTEPS": 1e7,
                "UPDATE_EPOCHS": 1,
                "NUM_MINIBATCHES": 1,
                "GAMMA": 0.998,
                "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.207,
                "ENT_COEF": 1.624e-07,
                "VF_COEF": 0.996,
                "MAX_GRAD_NORM": 0.5,
                "ANNEAL_LR": True,
                "DEBUG": True,
                
                "ENV_NAME": "alphatradeExec-v0",
                "NORMALIZE_ENV": False,
                "ENV_LENGTH": "oneWindow",
                "ATFOLDER": ATFolder,
                "TASKSIDE":'sell',
                "LAMBDA":0.0,
                "TASK_SIZE":500,
            }
        import flax
        from gymnax_exchange.jaxrl.gru_ppo_gera import ActorCriticRNN
        from gymnax_exchange.jaxrl.gru_ppo_gera import ScannedRNN
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=gru_config)
        init_hstate = ScannedRNN.initialize_carry(gru_config["NUM_ENVS"], 128)
            
        # ===================================================
        # CHOICE ONE
        with open(paramsFile_gru, 'rb') as f:
            restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"pramas restored")
        # ---------------------------------------------------
        # init_x = (
        #     jnp.zeros(
        #         (1, ppo_config["NUM_ENVS"], *env.observation_space(env_params).shape)
        #     ),
        #     jnp.zeros((1, ppo_config["NUM_ENVS"])),
        # )
        # network_params = network.init(key_policy, init_hstate, init_x)
        # restored_params = network_params
        # CHOICE OTWO
        # ===================================================
        
        
        init_done = jnp.array([False]*gru_config["NUM_ENVS"])
        ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
        assert len(ac_in[0].shape) == 3
        hstate, pi, value = network.apply(restored_params, init_hstate, ac_in)
        print("Network Carry Initialized")
        action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None) # CAUTION about the [0,0,:], only works for num_env=1
        print(f"-------------\nGRU 0th actions are: {action} with sum {action.sum()}")
        obs,state,reward,done,info=env.step(key_step, state, action, env_params)
        print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
        excuted_list = []
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from trained network ----------
            ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
            assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
            assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
            hstate, pi, value = network.apply(restored_params, hstate, ac_in) 
            action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
            # ---------- acion from trained network ----------
            # ==================== ACTION ====================    
            print(f"-------------\nGRU {i}th actions are: {action} with sum {action.sum()}")
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'],info['average_price'], info['slippage_rm'], info['price_adv_rm'], info['price_drift_rm'], info['advantage_reward'],info['total_revenue'],info['task_to_execute'] ,excuted_list
    
    # Time-weighted average price
    
    def get_twap_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        print("Time for reset: \n",time.time()-start)
        print(env_params.message_data.shape, env_params.book_data.shape)
        excuted_list = []
        for i in range(1,10000):
            # ==================== ACTION ====================
            # ---------- acion from given strategy  ----------
            print("---"*20)
            print("window_index ",state.window_index)
            key_policy, _ = jax.random.split(key_policy,2)
            
            def twapV3(state, env_params):
                # ---------- ifMarketOrder ----------
                remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
                marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
                ifMarketOrder = (remainingTime <= marketOrderTime)
                # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
                # ---------- ifMarketOrder ----------
                # ---------- quants ----------
                remainedQuant = state.task_to_execute - state.quant_executed
                remainedStep = state.max_steps_in_episode - state.step_counter
                stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
                limit_quants = jax.random.permutation(key_policy, jnp.array([stepQuant//2,stepQuant-stepQuant//2,stepQuant//2,stepQuant-stepQuant//2]), independent=True)
                market_quants = jnp.array([remainedQuant - 3*remainedQuant//4,remainedQuant//4, remainedQuant//4, remainedQuant//4])
                quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                # ---------- quants ----------
                return jnp.array(quants) 
            
            twap_action = twapV3(state, env_params)
            print(f"Sampled {i}th actions are: ",twap_action)
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,twap_action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("excuted ",info["quant_executed"])
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'],info['average_price'], info['slippage_rm'], info['price_adv_rm'], info['price_drift_rm'], info['advantage_reward'],info['total_revenue'],info['task_to_execute'] ,excuted_list
    
    # Random average price

    def get_random_average_price(rngInitNum):
        # ---------- init probabilities ----------
        import numpy as np
        p_0_1 = 0.9 # Define the probabilities for the numbers 0 and 1
        p_2_10 = 0.1 # Define the remaining probability for the numbers 2 through 10
        numbers_2_10 = np.arange(2, 200) # Define the numbers 2 through 10
        pareto_distribution = (1 / numbers_2_10)**4 # Generate a Pareto distribution for the numbers 2 through 200
        pareto_distribution /= pareto_distribution.sum()
        pareto_distribution *= p_2_10
        probabilities = np.array([p_0_1 / 2, p_0_1 / 2] + list(pareto_distribution)) # Combine the probabilities for all numbers from 0 to 10
        assert np.isclose(probabilities.sum(), 1.0) # Verify that the probabilities sum to 1
        # ---------- init probabilities ----------
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        print("Time for reset: \n",time.time()-start)
        excuted_list = []
        for i in range(1,10000):
            print("---"*20)
            print("window_index ",state.window_index)
            key_policy, _ = jax.random.split(key_policy,2)
            def randomV1(state, env_params):
                quants = np.random.choice(np.arange(0, 200), size=4, p=probabilities) # Generate random data from the custom distribution
                return jnp.array(quants)                
            random_action = randomV1(state, env_params)
            print(f"Sampled {i}th actions are: ",random_action)
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,random_action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("excuted ",info["quant_executed"])
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'],info['average_price'], info['slippage_rm'], info['price_adv_rm'], info['price_drift_rm'], info['advantage_reward'],info['total_revenue'],info['task_to_execute'] ,excuted_list
    
    # Hush average price

    def get_hush_average_price(rngInitNum):
        rng = jax.random.PRNGKey(rngInitNum)
        rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
        start=time.time()
        obs,state=env.reset(key_reset,env_params)
        print("Time for reset: \n",time.time()-start)
        excuted_list = []
        for i in range(1,10000):
            print("---"*20)
            print("window_index ",state.window_index)
            key_policy, _ = jax.random.split(key_policy,2)
            def rushV1(state, env_params):
                quants = np.random.choice(np.arange(0, 200), size=4) # Generate random data from the custom distribution
                return jnp.array(quants)                
            random_action = rushV1(state, env_params)
            print(f"Sampled {i}th actions are: ",random_action)
            start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,random_action, env_params)
            print(f"Time for {i} step: \n",time.time()-start)
            print("excuted ",info["quant_executed"])
            excuted_list.append(info["quant_executed"])
            if done:
                break
        return info['window_index'],info['average_price'], info['slippage_rm'], info['price_adv_rm'], info['price_drift_rm'], info['advantage_reward'],info['total_revenue'],info['task_to_execute'] ,excuted_list
    

    # def get_best_price_average_price(rngInitNum):
    #     rng = jax.random.PRNGKey(rngInitNum)
    #     rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    #     start=time.time()
    #     obs,state=env.reset(key_reset,env_params)
    #     print("Time for reset: \n",time.time()-start)
    #     excuted_list = []
    #     for i in range(1,10000):
    #         print("---"*20)
    #         print("window_index ",state.window_index)
    #         key_policy, _ = jax.random.split(key_policy,2)
    #         def bestPrice(state, env_params):
    #             quants = jnp.array([min(remaining Q, size at best price), 0, 0, 0])
    #             return jnp.array(quants)                
    #         random_action = bestPrice(state, env_params)
    #         print(f"Sampled {i}th actions are: ",random_action)
    #         start=time.time()
    #         obs,state,reward,done,info=env.step(key_step, state,random_action, env_params)
    #         print(f"Time for {i} step: \n",time.time()-start)
    #         print("excuted ",info["quant_executed"])
    #         excuted_list.append(info["quant_executed"])
    #         if done:
    #             break
    #     return info['window_index'], info['average_price'], excuted_list


    def get_advantage(rngInitNum):
        window_index2,twap,twap_slip,twap_adv,twap_drift,twap_reward,twap_totalrev,twap_taskex,executed_list2=get_twap_average_price(rngInitNum)
        window_index1,ppo,ppo_slip,ppo_adv,ppo_drift,ppo_reward,ppo_totalrev,ppo_taskex,executed_list1=get_ppo_average_price(rngInitNum)
        window_index3,random,random_slip,random_adv,random_drift,random_reward,random_totalrev,random_taskex,executed_list3=get_random_average_price(rngInitNum)
        window_index4,rush,rush_slip,rush_adv,rush_drift,rush_reward,rush_totalrev,rush_taskex,executed_list4=get_hush_average_price(rngInitNum)
        window_index5,lstm,lstm_slip,lstm_adv,lstm_drift,lstm_reward,lstm_totalrev,lstm_taskex,executed_list5=get_ppo_lstm_average_price(rngInitNum)
        window_index6,gru,gru_slip,gru_adv,gru_drift,gru_reward,gru_totalrev,gru_taskex,executed_list6=get_ppo_gru_average_price(rngInitNum)
        window_index7, ac,ac_slip,ac_adv,ac_drift,ac_reward,ac_totalrev,ac_taskex,executed_list7 = ac_model.execute_ac_strategy(trade_list, rngInitNum)
        assert window_index1 == window_index2
        assert window_index1 == window_index3
        assert window_index1 == window_index4
        assert window_index1 == window_index5
        assert window_index1 == window_index6
        assert window_index1 == window_index7
        return window_index1, (ppo-twap)/twap*10000, (ppo-random)/random*10000, (ppo-rush)/rush*10000,(ppo-ac)/ac*10000,(lstm-twap)/twap*10000, (lstm-random)/random*10000, (lstm-rush)/rush*10000,(lstm-ac)/ac*10000, (gru-twap)/twap*10000,(gru-random)/random*10000,(gru-rush)/rush*10000,(gru-ac)/gru*10000, ppo, lstm, gru, twap, random, rush, ac, twap_slip, ppo_slip, random_slip , rush_slip, lstm_slip, gru_slip, ac_slip, twap_adv, ppo_adv, random_adv , rush_adv, lstm_adv, gru_adv, ac_adv, twap_drift, ppo_drift, random_drift , rush_drift, lstm_drift, gru_drift, ac_drift, executed_list1, executed_list2, executed_list3, executed_list4, executed_list5, executed_list6, executed_list7, twap_reward, ppo_reward,random_reward, rush_reward, lstm_reward, gru_reward,ac_reward, twap_totalrev, ppo_totalrev,random_totalrev , rush_totalrev,lstm_totalrev,gru_totalrev,ac_totalrev ,twap_taskex,ppo_taskex,random_taskex,rush_taskex,lstm_taskex,gru_taskex,ac_taskex

    # result_list = []
    for rngInitNum in range(100,500):
        print(f"++++ rngInitNum {rngInitNum}")
        result_tuple = get_advantage(rngInitNum) 
        # result_list.append(result_tuple[0]) # window index
        # result_tuple = tuple(x.item() if hasattr(x, 'item') else x for x in result_tuple)
        print(f"window_index {result_tuple[0]:<4} , advantageTWAP {result_tuple[1]:^20} , advantageRANDOM {result_tuple[2]:^20} , advantageRUSH {result_tuple[3]:^20} , advantageAC  {result_tuple[4]:^20} ,lstmTWAP {result_tuple[5]:^20} , lstmRANDOM {result_tuple[6]:^20} , lstmRUSH {result_tuple[7]:^20} , lstmAC {result_tuple[8]:^20}, gruTWAP {result_tuple[9]:^20} , gruRANDOM {result_tuple[10]:^20} , gruRUSH {result_tuple[11]:^20} , gruAC {result_tuple[12]:^20}, ppoAP {result_tuple[13]:<20} , lstmAP {result_tuple[14]:<20} , gruAP {result_tuple[15]:<20} ,  twapAP {result_tuple[16]:<20} , randomAP {result_tuple[17]:<20} , rushAP {result_tuple[18]:<20} , acAP {result_tuple[19]:^20}, twap_slip {result_tuple[20]:<20}, ppo_slip {result_tuple[21]:<20}, random_slip {result_tuple[22]:<20} , rush_slip {result_tuple[23]:<20}, lstm_slip {result_tuple[24]:<20}, gru_slip {result_tuple[25]:<20}, ac_slip {result_tuple[26]:<20}, twap_adv {result_tuple[27]:<20}, ppo_adv {result_tuple[28]:<20}, random_adv {result_tuple[29]:<20}, rush_adv {result_tuple[30]:<20}, lstm_adv {result_tuple[31]:<20}, gru_adv {result_tuple[32]:<20}, ac_adv {result_tuple[33]:<20}, twap_drift {result_tuple[34]:<20}, ppo_drift {result_tuple[35]:<20}, random_drift {result_tuple[36]:<20}, rush_drift {result_tuple[37]:<20}, lstm_drift {result_tuple[38]:<20}, gru_drift {result_tuple[39]:<20}, ac_drift {result_tuple[40]:<20}, ppoExecuted { [int(x) for x in result_tuple[41]]} , twapExecuted { [int(x) for x in result_tuple[42]]} , randomExecuted { [int(x) for x in result_tuple[43]]} , rushExecuted { [int(x) for x in result_tuple[44]]}, lstmExecuted { [int(x) for x in result_tuple[45]]}, gruExecuted { [int(x) for x in result_tuple[46]]}, acExecuted { [int(x) for x in result_tuple[47]]},twap_reward {result_tuple[48]:<20}, ppo_reward {result_tuple[49]:<20}, random_reward {result_tuple[50]:<20}, rush_reward {result_tuple[51]:<20}, lstm_reward {result_tuple[52]:<20}, gru_reward {result_tuple[53]:<20}, ac_reward {result_tuple[54]:<20}, twap_totalrev {result_tuple[55]:<20}, ppo_totalrev {result_tuple[56]:<20}, random_totalrev {result_tuple[57]:<20}, rush_totalrev {result_tuple[58]:<20}, lstm_totalrev {result_tuple[59]:<20}, gru_totalrev {result_tuple[60]:<20}, ac_totalrev {result_tuple[61]:<20}, twap_taskex {result_tuple[62]:<20}, ppo_taskex {result_tuple[63]:<20}, random_taskex {result_tuple[64]:<20}, rush_taskex {result_tuple[65]:<20}, lstm_taskex {result_tuple[66]:<20}, gru_taskex {result_tuple[67]:<20}, ac_taskex {result_tuple[68]:<20}",\
            file=open(outputfile,'a'))
    
    
    
    


