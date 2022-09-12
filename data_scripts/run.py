import sys
import os

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


config = [
    {
        'dataset': 'slc/space',
        'train': 'train[:80%]',
        'test': 'train[80%:]',
        'eval': 'test',
        'batch_size': 50,
        'ensemble_epochs': 100,
        'subset_train_to': None,
        'scaling': None,
        'builder_kwargs': None,
    },
    {
        'dataset': 'mirabest/confident',
        'train': 'train[:80%]',
        'test': 'train[80%:]',
        'eval': 'test',
        'batch_size': 53,
        'ensemble_epochs': 100,
        'subset_train_to': None,
        'scaling': None,
        'builder_kwargs': None,
    }
]

model = 'lenet'
num_repeats = 3
num_ensemble_repeats = 5
image_size = 64

for c in config:

    cmd_args = get_vi_args()
    train_utils.set_up_jax(cmd_args.tpu_ip, cmd_args.use_float64)
    script_utils.print_visible_devices()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = c['subset_train_to']
    cmd_args.model_name = model
    cmd_args.train_split = c['train']
    cmd_args.test_split = c['test']

    # VI

    print('Performing VI')

    cmd_args.weight_decay = 5
    cmd_args.init_step_size = 1e-4
    cmd_args.num_epochs = 200
    cmd_args.eval_freq = 5
    cmd_args.batch_size = c['batch_size']
    cmd_args.patience = 10
    cmd_args.save_freq = 20
    cmd_args.optimizer = 'Adam'
    cmd_args.vi_sigma_init = 0.01
    cmd_args.vi_ensemble_size = 20

    for i in range(num_repeats):
        cmd_args.eval_split = None
        cmd_args.seed = i
        cmd_args.dir = 'runs/vi/%s/%s/' % (c['dataset'], i)
        train_vi_model(cmd_args)
        cmd_args.eval_split = c['eval']
        train_vi_model(cmd_args)

    # SGD

    print('Performing Deep Ensembles')

    cmd_args = get_sgd_args()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = c['subset_train_to']
    cmd_args.model_name = model
    cmd_args.train_split = c['train']
    cmd_args.test_split = c['test']

    cmd_args.weight_decay = 10
    cmd_args.init_step_size = 3e-7
    cmd_args.num_epochs = c['ensemble_epochs']
    cmd_args.batch_size = c['batch_size']
    cmd_args.eval_freq = 5
    cmd_args.save_freq = 20
    cmd_args.optimizer = 'SGD'

    for i in range(num_repeats):
        cmd_args.ensemble_root = None
        for j in range(num_ensemble_repeats):
            cmd_args.eval_split = None
            cmd_args.seed = i*num_ensemble_repeats + j
            cmd_args.dir = 'runs/sgd/%s/%s/%s/' % (c['dataset'], i, j)
            train_sgd_model(cmd_args)
            cmd_args.eval_split = c['eval']
            train_sgd_model(cmd_args)
        cmd_args.ensemble_root = 'runs/sgd/%s/%s/*/*' % (c['dataset'], i)
        train_sgd_model(cmd_args)

    # HMC

    print('Performing HMC')

    cmd_args = get_hmc_args()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = c['subset_train_to']
    cmd_args.model_name = model
    cmd_args.train_split = c['train']
    # Compute ensemble predictions directly as we don't use early stopping
    cmd_args.test_split = c['eval']

    cmd_args.weight_decay = 50
    cmd_args.temperature = 1.0
    cmd_args.step_size = 3.0e-5
    cmd_args.trajectory_len = 0.1
    cmd_args.num_iterations = 50
    cmd_args.max_num_leapfrog_steps = 10000
    cmd_args.num_burn_in_iterations = 10

    for i in range(num_repeats):
        cmd_args.eval_split = None
        cmd_args.seed = i
        cmd_args.dir = 'runs/hmc/%s/%s/' % (c['dataset'], i)
        train_hmc_model(cmd_args)
