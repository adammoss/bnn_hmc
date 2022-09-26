import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from scripts.run_sgd import train_model as train_sgd_model
from scripts.run_sgd import get_args as get_sgd_args
from data_scripts.data_config import config, model, image_size, num_repeats, num_ensemble_repeats

print('Performing MCD')

for c in config:

    cmd_args = get_sgd_args()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = c['subset_train_to']
    cmd_args.model_name = model + '_dropout'
    cmd_args.train_split = c['train']
    cmd_args.test_split = c['test']

    cmd_args.weight_decay = 10
    cmd_args.batch_size = c['batch_size']
    cmd_args.num_epochs = 200
    cmd_args.patience = 10
    cmd_args.eval_freq = 5
    cmd_args.save_freq = 20
    cmd_args.dropout_rate = 0.1
    cmd_args.repeats = num_ensemble_repeats
    if c['optimizer'] == 'SGD':
        cmd_args.init_step_size = 3e-7
    else:
        cmd_args.init_step_size = 1e-5
    cmd_args.optimizer = c['optimizer']

    for i in range(num_repeats):
        cmd_args.eval_split = None
        cmd_args.seed = i
        cmd_args.dir = 'runs/mcd/%s/%s/' % (c['dataset'], i)
        train_sgd_model(cmd_args)
        cmd_args.eval_split = c['eval']
        train_sgd_model(cmd_args)

    for ood in c['ood']:
        for i in range(num_repeats):
            output_prefix = 'ood_%s_%s' % (ood['dataset'], ood['eval'])
            cmd_args.output_prefix = output_prefix.replace('/', '_')
            cmd_args.dataset_name = ood['dataset']
            cmd_args.builder_kwargs = ood['builder_kwargs']
            cmd_args.eval_split = ood['eval']
            cmd_args.seed = i
            cmd_args.dir = 'runs/mcd/%s/%s/' % (c['dataset'], i)
            train_sgd_model(cmd_args)
