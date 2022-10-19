import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from scripts.run_sgd import train_model as train_sgd_model
from scripts.run_sgd import get_args as get_sgd_args
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import script_utils
from data_scripts.data_config import config, image_size, num_repeats, num_ensemble_repeats, subset_train_to

print('Performing MCD')

for c in config:

    cmd_args = get_sgd_args()
    train_utils.set_up_jax(cmd_args.tpu_ip, cmd_args.use_float64)
    script_utils.print_visible_devices()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = subset_train_to
    cmd_args.model_name = c['model'] + '_dropout'
    cmd_args.train_split = c['train']
    cmd_args.test_split = c['test']

    cmd_args.weight_decay = 10
    cmd_args.batch_size = c['batch_size']
    cmd_args.num_epochs = 300
    cmd_args.patience = 50
    cmd_args.eval_freq = 5
    cmd_args.save_freq = 20
    cmd_args.dropout_rate = 0.1
    cmd_args.repeats = num_ensemble_repeats
    cmd_args.init_step_size = c['step_size']
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
