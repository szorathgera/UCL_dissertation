import os
import sys
import time
import datetime
import flax
import jax
import chex
import flax.linen as nn
import jax.numpy as jnp
import optax
import numpy as np
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
from typing import Any, Dict, NamedTuple, Sequence
import distrax
from jax import config
from gymnax.environments import spaces
import gymnax

# Configure JAX
os.environ['JAX_PLATFORMS'] = 'cpu'
config.update("jax_disable_jit", False)
config.update("jax_check_tracer_leaks", False)

# Set up paths
sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')

# Import custom modules
from purejaxrl.experimental.s5.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from purejaxrl.experimental.s5.wrappers import FlattenObservationWrapper, LogWrapper
from gymnax_exchange.jaxen.exec_env import ExecutionEnv

# Set up WandB
wandbOn = True
if wandbOn:
    import wandb

# Define paths for results and checkpoints
timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
results_file_path = os.path.join('/content/drive/My Drive/YourProject/results', f"results_file_{timestamp}")
checkpoints_dir_path = os.path.join('/content/drive/My Drive/YourProject/checkpoints', f"checkpoints_{timestamp}")

os.makedirs(results_file_path, exist_ok=True)
os.makedirs(checkpoints_dir_path, exist_ok=True)

# Function to save checkpoints
def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"Checkpoint saved to {filename}")

# S5 model parameters
d_model = 256
ssm_size = 256
C_init = "lecun_normal"
discretization = "zoh"
dt_min = 0.001
dt_max = 0.1
n_layers = 4
conj_sym = True
clip_eigs = False
bidirectional = False

blocks = 1
block_size = int(ssm_size / blocks)
Lambda, _, B, V, B_orig = make_DPLR_HiPPO(ssm_size)
block_size = block_size // 2
ssm_size = ssm_size // 2
Lambda = Lambda[:block_size]
V = V[:, :block_size]
Vinv = V.conj().T

ssm_init_fn = init_S5SSM(H=d_model,
                         P=ssm_size,
                         Lambda_re_init=Lambda.real,
                         Lambda_im_init=Lambda.imag,
                         V=V,
                         Vinv=Vinv,
                         C_init=C_init,
                         discretization=discretization,
                         dt_min=dt_min,
                         dt_max=dt_max,
                         conj_sym=conj_sym,
                         clip_eigs=clip_eigs,
                         bidirectional=bidirectional)

class ActorCriticS5(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.encoder_0 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.encoder_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.action_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_decoder = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.5))
        self.value_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_decoder = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.s5 = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=d_model,
            n_layers=n_layers,
            activation="half_glu1",
        )
        self.actor_logtstd = self.param("log_std", nn.initializers.constant(-0.7), (self.action_dim,))

    def __call__(self, hidden, x):
        obs, dones = x
        embedding = self.encoder_0(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = self.encoder_1(embedding)
        embedding = nn.leaky_relu(embedding)
        hidden, embedding = self.s5(hidden, embedding, dones)
        actor_mean = self.action_body_0(embedding)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_body_1(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_decoder(actor_mean)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.actor_logtstd))
        critic = self.value_body_0(embedding)
        critic = nn.leaky_relu(critic)
        critic = self.value_body_1(critic)
        critic = nn.leaky_relu(critic)
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
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    env = ExecutionEnv(config["ATFOLDER"], config["TASKSIDE"], config["WINDOW_INDEX"], config["ACTION_TYPE"], config["TASK_SIZE"], config["REWARD_LAMBDA"])
    env_params = env.default_params
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCriticS5(env.action_space(env_params).shape[0], config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
                  jnp.zeros((1, config["NUM_ENVS"])))
        init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
        network_params = network.init(_rng, init_hstate, init_x)
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                         optax.adam(learning_rate=linear_schedule, b1=0.9, b2=0.99, eps=1e-5) if config["ANNEAL_LR"] else optax.adam(config["LR"], b1=0.9, b2=0.99, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(params, init_hstate, (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0), shuffled_batch)
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            init_hstate = initial_hstate
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = (traj_batch.info, train_state.params)
            rng = update_state[-1]
            if config.get("DEBUG"):
                def callback(metric):
                    info, trainstate_params = metric
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]

                    def evaluation():
                        if not os.path.exists(config['CHECKPOINT_DIR']): os.makedirs(config['CHECKPOINT_DIR'])
                        if any(timesteps % int(1e3) == 0) and len(timesteps) > 0:
                            start = time.time()
                            jax.debug.print(">>> checkpoint saving {}", round(timesteps[0], -3))
                            checkpoint_filename = os.path.join(config['CHECKPOINT_DIR'], f"checkpoint_{round(timesteps[0], -3)}.ckpt")
                            save_checkpoint(trainstate_params, checkpoint_filename)
                            jax.debug.print("+++ checkpoint saved  {}", round(timesteps[0], -3))
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
                            wandb.log({"global_step": timesteps[t], "episodic_return": return_values[t], "episodic_revenue": revenues[t], "quant_executed": quant_executed[t], "average_price": average_price[t], "slippage_rm": slippage_rm[t], "price_adv_rm": price_adv_rm[t], "price_drift_rm": price_drift_rm[t], "vwap_rm": vwap_rm[t], "current_step": current_step[t], "advantage_reward": advantage_reward[t]})
                            print(f"global step={timesteps[t]:<8} | episodic return={return_values[t]:<15} | episodic revenue={revenues[t]:<15} | average_price={average_price[t]:<20}", file=open(config['RESULTS_FILE'], 'a'))
                        else:
                            print(f"global step={timesteps[t]:<8} | episodic return={return_values[t]:<15} | episodic revenue={revenues[t]:<15} | average_price={average_price[t]:<20}")
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")

    ppo_config = {
        "LR": 2.5e-5,
        "ENT_COEF": 0.1,
        "NUM_ENVS": 500,
        "TOTAL_TIMESTEPS": 1e8,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 5,
        "NUM_STEPS": 455,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": False,
        "ACTOR_TYPE": "S5",
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": -1,
        "DEBUG": True,
        "ATFOLDER": ".",
        "TASKSIDE": 'sell',
        "REWARD_LAMBDA": 1,
        "ACTION_TYPE": "pure",
        "TASK_SIZE": 500,
        "RESULTS_FILE": os.path.join(results_file_path, f"results_file_{timestamp}"),
        "CHECKPOINT_DIR": os.path.join(checkpoints_dir_path, f"checkpoints_{timestamp}")
    }

    if wandbOn:
        run = wandb.init(project="AlphaTradeJAX_Train", config=ppo_config, save_code=True)
        import datetime
        params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
        print(f"Results would be saved to {params_file_name}")
    else:
        params_file_name = f'params_file_{timestamp}'
        print(f"Results would be saved to {params_file_name}")

    rng = jax.random.PRNGKey(0)
    train_jit = jax.jit(make_train(ppo_config))
    start = time.time()
    out = train_jit(rng)
    print("Time: ", time.time() - start)

    train_state = out['runner_state'][0]
    params = train_state.params

    with open(params_file_name, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"params saved")

    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"params restored")

    if wandbOn:
        run.finish()

