import os


os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'



from jaxlib import xla_extension
import jax
import jax.numpy as jnp
from jaxlib import xla_extension

print(dir(xla_extension))
print("Num Jax Devices:", jax.device_count(), "Device List:", jax.devices())




def main():
    print("Devices:", jax.devices())

    x = jnp.arange(10)
    print(x)

if __name__ == "__main__":
    main()

import os
import datetime
import flax
import time
from google.colab import drive

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Define base and timestamped paths
base_checkpoints_dir = '/content/drive/My Drive/YourProject/checkpoints'
timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
timestamped_checkpoints_dir = os.path.join(base_checkpoints_dir, f"checkpoints_{timestamp}")

# Ensure the base and timestamped directories exist
os.makedirs(base_checkpoints_dir, exist_ok=True)
os.makedirs(timestamped_checkpoints_dir, exist_ok=True)

# Step 3: Save Checkpoint Function
def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"Checkpoint saved to {filename}")

# Example Callback Function
def callback(metric):
    info, trainstate_params = metric
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
    
    def evaluation():
        if not os.path.exists(timestamped_checkpoints_dir):
            os.makedirs(timestamped_checkpoints_dir)
        # Inside your loop or function where you save the checkpoint
        if any(timesteps % int(1e3) == 0) and len(timesteps) > 0:  # +1 since global_step is 0-indexed
            start = time.time()
            jax.debug.print(">>> checkpoint saving {}", round(timesteps[0], -3))
            # Save the checkpoint to the specific directory
            checkpoint_filename = os.path.join(timestamped_checkpoints_dir, f"checkpoint_{round(timesteps[0], -3)}.ckpt")
            save_checkpoint(trainstate_params, checkpoint_filename)  # Assuming trainstate_params contains your model's state
            jax.debug.print("+++ checkpoint saved  {}", round(timesteps[0], -3))
            jax.debug.print("+++ time taken        {}", time.time()-start)
    
    evaluation()

# Example usage within your training loop
for epoch in range(5):  # Example loop, replace with your actual training loop
    # Simulate timesteps and trainstate_params
    timesteps = [1000 * (epoch + 1)]  # Simulated timesteps
    trainstate_params = {"param": epoch}  # Simulated parameters
    
    checkpoint_filename = os.path.join(timestamped_checkpoints_dir, f"checkpoint_{round(timesteps[0], -3)}.ckpt")
    save_checkpoint(trainstate_params, checkpoint_filename)

    results_file = os.path.join(timestamped_checkpoints_dir, f"results_file_{timestamp}.txt")
    with open(results_file, 'a') as f:
        f.write(f"Epoch {epoch}, Timesteps {timesteps[0]}, Params: {trainstate_params}\n")
        print(f"Results saved to {results_file}")

# Confirming directory contents (optional)
print("Checkpoints:", os.listdir(timestamped_checkpoints_dir))