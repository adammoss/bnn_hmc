import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from scripts.run_sgd import train_model as train_sgd_model
from scripts.run_sgd import get_args as get_sgd_args

config = [
    {
        'dataset': 'mirabest/confident',
        'model': 'lenet',
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

num_repeats = 3
num_ensemble_repeats = 10
image_size = 64

for c in config:

    # MCD

    print('Performing MCD')

    cmd_args = get_sgd_args()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = c['subset_train_to']
    cmd_args.model_name = c['model'] + '_dropout'
    cmd_args.train_split = c['train']
    cmd_args.test_split = c['test']

    cmd_args.weight_decay = 10
    cmd_args.init_step_size = 3e-7
    cmd_args.batch_size = c['batch_size']
    cmd_args.num_epochs = 200
    cmd_args.patience = 10
    cmd_args.eval_freq = 5
    cmd_args.save_freq = 5
    cmd_args.dropout_rate = 0.1
    cmd_args.repeats = num_ensemble_repeats
    cmd_args.optimizer = 'SGD'

    for i in range(num_repeats):
        cmd_args.eval_split = None
        cmd_args.seed = i
        cmd_args.dir = 'runs/mcd/%s/%s/' % (c['dataset'], i)
        train_sgd_model(cmd_args)
        cmd_args.eval_split = c['eval']
        train_sgd_model(cmd_args)
