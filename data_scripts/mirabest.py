import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from scripts.run_sgd import train_model as train_sgd_model
from scripts.run_hmc import train_model as train_hmc_model
from scripts.run_vi import train_model as train_vi_model
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import script_utils

cmd_args = {
    'tpu_ip': None,
    'use_float64': False,
    'seed': 0,
    'weight_decay': 10,
    'dir': 'results/sgd/mirabestc/0/',
    'dataset_name': 'mirabest/confident',
    'model_name': 'lenet',
    'init_step_size': 3e-7,
    'num_epochs': 100,
    'eval_freq': 5,
    'batch_size': 53,
    'save_freq': 5,
    'optimzer': 'SGD',
    'image_size': 64,
    'train_split': 'train[:80%]',
    'test_split': 'train[80%:]',
}

train_utils.set_up_jax(cmd_args.tpu_ip, cmd_args.use_float64)
script_utils.print_visible_devices()
train_sgd_model(cmd_args)