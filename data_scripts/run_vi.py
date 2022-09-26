import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from scripts.run_vi import train_model as train_vi_model
from scripts.run_vi import get_args as get_vi_args
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import script_utils
from data_scripts.data_config import config, model, image_size, num_repeats, num_ensemble_repeats

print('Performing VI')

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

    cmd_args.weight_decay = 5
    cmd_args.init_step_size = 1e-4
    cmd_args.num_epochs = 300
    cmd_args.eval_freq = 5
    cmd_args.batch_size = c['batch_size']
    cmd_args.patience = 20
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

    for ood in c['ood']:
        for i in range(num_repeats):
            output_prefix = 'ood_%s_%s' % (ood['dataset'], ood['eval'])
            cmd_args.output_prefix = output_prefix.replace('/', '_')
            cmd_args.dataset_name = ood['dataset']
            cmd_args.builder_kwargs = ood['builder_kwargs']
            cmd_args.eval_split = ood['eval']
            cmd_args.seed = i
            cmd_args.dir = 'runs/vi/%s/%s/' % (c['dataset'], i)
            train_vi_model(cmd_args)
