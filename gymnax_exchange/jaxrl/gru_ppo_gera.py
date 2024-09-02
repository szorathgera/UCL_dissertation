import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex

# Assume the paths are correctly set
sys.path.append('/Users/szorathgera/dissertation/purejaxrl')
sys.path.append('/Users/szorathgera/dissertation/AlphaTrade')
from purejaxrl.purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper, ClipAction, VecEnv, NormalizeVecObservation, NormalizeVecReward
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
import flax

from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
print(jax.devices()[0]) 

# Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
config.update("jax_check_tracer_leaks", True) 
import datetime

wandbOn = True
if wandbOn:
    import wandb

def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"Checkpoint saved to {filename}")



class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        
        initial_carry = self.initialize_carry(ins.shape[0], rnn_state[0].shape[-1])
        #jax.debug.print("Shapes - resets: {}, rnn_state: {}, initial_carry: {}", resets.shape, rnn_state[0].shape, initial_carry[0].shape)

        rnn_state = jax.tree_util.tree_map(lambda s, c: jnp.where(resets[:, None], c, s), rnn_state, initial_carry)

        new_rnn_state, y = nn.GRUCell(features=rnn_state[0].shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.encoder0 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.action_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_decoder = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.5))
        
        self.value_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_decoder = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        
        self.rnn = ScannedRNN()
        self.log_std = self.param("log_std", nn.initializers.constant(-0.7), (self.action_dim,))

    def __call__(self, hidden, x):
        obs, dones = x
        embedding = self.encoder0(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = self.rnn(hidden, rnn_in)

        actor_mean = self.action_body_0(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = self.action_decoder(actor_mean)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std))

        critic = self.value_body_0(embedding)
        critic = nn.relu(critic)
        critic = self.value_decoder(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = ExecutionEnv(config["ATFOLDER"], config["TASKSIDE"], config["WINDOW_INDEX"], config["ACTION_TYPE"], config["TASK_SIZE"], config["REWARD_LAMBDA"])
    env_params = env.default_params
    env = LogWrapper(env)    
    
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, b1=0.9, b2=0.99, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], b1=0.9, b2=0.99, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done_step, action, value, reward_step, log_prob, last_obs, info_step
                )
                runner_state = (train_state, env_state_step, obsv_step, done_step, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value, next_done = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None,:] # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = (traj_batch.info, train_state.params)
            rng = update_state[-1]
            if config.get("DEBUG"):
                def callback(metric):
                    info, trainstate_params = metric
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )

                    def round_to_nearest_thousand(x):
                        return jnp.round(x / 1000) * 1000

                    def evaluation():
                        if not os.path.exists(config['CHECKPOINT_DIR']):
                            os.makedirs(config['CHECKPOINT_DIR'])
                        if any(timesteps % int(1e3) == 0) and len(timesteps) > 0:
                            start = time.time()
                            jax.debug.print(">>> checkpoint saving {}", round_to_nearest_thousand(timesteps[0]))
                            checkpoint_filename = os.path.join(config['CHECKPOINT_DIR'], f"checkpoint_{round_to_nearest_thousand(timesteps[0])}.ckpt")
                            save_checkpoint(trainstate_params, checkpoint_filename)
                            jax.debug.print("+++ checkpoint saved  {}", round_to_nearest_thousand(timesteps[0]))
                            jax.debug.print("+++ time taken        {}", time.time() - start)
                    evaluation()

                    revenues = info["total_revenue"][info["returned_episode"]]
                    quant_executed = info["quant_executed"][info["returned_episode"]]
                    average_price = info["average_price"][info["returned_episode"]]
                    slippage_rm = info["slippage_rm"][info["returned_episode"]]
                    price_drift_rm = info["price_drift_rm"][info["returned_episode"]]
                    price_adv_rm = info["price_adv_rm"][info["returned_episode"]]
                    vwap_rm = info["vwap_rm"][info["returned_episode"]]
                    current_step = info["current_step"][info["returned_episode"]]
                    advantage_reward = info["advantage_reward"][info["returned_episode"]]

                    for t in range(len(timesteps)):
                        if wandbOn:
                            wandb.log(
                                {
                                    "global_step": timesteps[t],
                                    "episodic_return": return_values[t],
                                    "episodic_revenue": revenues[t],
                                    "quant_executed": quant_executed[t],
                                    "average_price": average_price[t],
                                    "slippage_rm": slippage_rm[t],
                                    "price_adv_rm": price_adv_rm[t],
                                    "price_drift_rm": price_drift_rm[t],
                                    "vwap_rm": vwap_rm[t],
                                    "current_step": current_step[t],
                                    "advantage_reward": advantage_reward[t],
                                }
                            )
                            print(
                                f"global step={timesteps[t]:<11} | episodic return={return_values[t]:<11} | episodic revenue={revenues[t]:<11} | average_price={average_price[t]:<11}",\
                                file=open(config['RESULTS_FILE'],'a')
                            )
                        else:
                            print(
                                f"global step={timesteps[t]:<8} | episodic return={return_values[t]:<15} | episodic revenue={revenues[t]:<15} | average_price={average_price[t]:<20}"
                            )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")

    ppo_config = {
        "LR": 1.116e-5,
        "ENT_COEF": 1.624e-07,
        "NUM_ENVS": 500,
        "TOTAL_TIMESTEPS": 1e7,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 5,
        "NUM_STEPS": 455,
        "CLIP_EPS": 0.207,
        "GAMMA": 0.998,
        "GAE_LAMBDA": 0.95,
        "VF_COEF": 0.996,
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": False,
        "ACTOR_TYPE": "RNN",
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": -1,
        "DEBUG": True,
        "ATFOLDER": "/Users/szorathgera/dissertation/AlphaTrade",
        "TASKSIDE": 'sell',
        "REWARD_LAMBDA": 1,
        "ACTION_TYPE": "pure",
        "TASK_SIZE": 500,
        "RESULTS_FILE": f"/Users/szorathgera/dissertation/AlphaTrade/gymnax_exchange/jaxrl/gru_ppo_train/results_file_{timestamp}",
        "CHECKPOINT_DIR": f"/Users/szorathgera/dissertation/AlphaTrade/gymnax_exchange/jaxrl/gru_ppo_train/checkpoints/checkpoints_{timestamp}",
    }

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=ppo_config,
            save_code=True,
        )
        import datetime
        params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
        print(f"Results would be saved to {params_file_name}")
    else:
        import datetime
        params_file_name = f'params_file_{timestamp}'
        print(f"Results would be saved to {params_file_name}")

    rng = jax.random.PRNGKey(0)
    train_jit = jax.jit(make_train(ppo_config))
    start = time.time()
    out = train_jit(rng)
    print("Time: ", time.time() - start)

    train_state = out['runner_state'][0]
    params = train_state.params

    import datetime
    params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'

    with open(params_file_name, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"pramas saved")

    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"pramas restored")

    if wandbOn:
        run.finish()
