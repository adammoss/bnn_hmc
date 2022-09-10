import sys
import os
from argparse import Namespace

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from scripts.run_sgd import train_model as train_sgd_model
from scripts.run_sgd import get_args as get_sgd_args
from scripts.run_hmc import train_model as train_hmc_model
from scripts.run_hmc import get_args as get_hmc_args
from scripts.run_vi import train_model as train_vi_model
from scripts.run_vi import get_args as get_vi_args
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import script_utils

dataset_name = 'mirabest/confident'
model_name = 'lenet'
num_repeats = 5

cmd_args = get_vi_args()
train_utils.set_up_jax(cmd_args.tpu_ip, cmd_args.use_float64)
script_utils.print_visible_devices()

cmd_args.dataset_name = dataset_name
cmd_args.model_name = model_name
cmd_args.train_split = 'train[:80%]'
cmd_args.test_split = 'train[80%:]'

# VI

cmd_args.weight_decay = 5
cmd_args.init_step_size = 1e-4
cmd_args.num_epochs = 200
cmd_args.eval_freq = 5
cmd_args.batch_size = 53
cmd_args.patience = 10
cmd_args.save_freq = 300
cmd_args.optimizer = 'Adam'
cmd_args.vi_sigma_init = 0.01
cmd_args.vi_ensemble_size = 20

for i in range(num_repeats):
    cmd_args.seed = i
    cmd_args.dir = 'runs/vi/mirabestc/%s/' % i
    train_vi_model(cmd_args)
