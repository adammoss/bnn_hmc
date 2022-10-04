import sys
import os
import glob

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from scripts.make_posterior_surface_plot import run_visualization
from scripts.make_posterior_surface_plot import get_args as get_visualization_args
from scripts.run_hmc import train_model as train_hmc_model
from scripts.run_hmc import get_args as get_hmc_args
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import script_utils
from data_scripts.data_config import config, model, image_size, num_repeats, num_ensemble_repeats, subset_train_to

print('Performing HMC')

for c in config:

    cmd_args = get_hmc_args()
    train_utils.set_up_jax(cmd_args.tpu_ip, cmd_args.use_float64)
    script_utils.print_visible_devices()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = subset_train_to
    cmd_args.model_name = model
    cmd_args.train_split = c['train']
    # Compute ensemble predictions directly as we don't use early stopping
    cmd_args.test_split = c['eval']

    cmd_args.weight_decay = 50
    cmd_args.temperature = 1.0
    cmd_args.step_size = 1.0e-5
    cmd_args.trajectory_len = 0.1
    cmd_args.num_iterations = 50
    cmd_args.max_num_leapfrog_steps = 50000
    cmd_args.num_burn_in_iterations = 10

    for ood in c['ood']:
        for i in range(num_repeats):
            cmd_args.seed = i
            cmd_args.dir = 'runs/hmc/%s/%s/' % (c['dataset'], i)
            cmd_args.test_dataset_name = ood['dataset']
            cmd_args.test_builder_kwargs = ood['builder_kwargs']
            train_hmc_model(cmd_args)

    for i in range(num_repeats):
        cmd_args.eval_split = None
        cmd_args.seed = i
        cmd_args.dir = 'runs/hmc/%s/%s/' % (c['dataset'], i)
        train_hmc_model(cmd_args)

    cmd_args = get_visualization_args()

    cmd_args.dataset_name = c['dataset']
    cmd_args.image_size = image_size
    cmd_args.builder_kwargs = c['builder_kwargs']
    cmd_args.scaling = c['scaling']
    cmd_args.subset_train_to = subset_train_to
    cmd_args.model_name = model
    cmd_args.train_split = c['train']
    cmd_args.eval_split = c['eval']

    cmd_args.weight_decay = 50
    cmd_args.temperature = 1.0

    for i in range(num_repeats):
        checkpoint1 = None
        checkpoint2 = None
        checkpoint3 = None
        cmd_args.dir = 'runs/hmc/%s/%s/' % (c['dataset'], i)
        for filename in glob.glob('runs/hmc/%s/%s/*/*.pt' % (c['dataset'], i)):
            if '_10.pt' in filename:
                checkpoint1 = filename
            if '_25.pt' in filename:
                checkpoint2 = filename
            if '_40.pt' in filename:
                checkpoint3 = filename
        if checkpoint1 is not None and checkpoint2 is not None and checkpoint3 is not None:
            cmd_args.checkpoint1 = checkpoint1
            cmd_args.checkpoint2 = checkpoint2
            cmd_args.checkpoint3 = checkpoint3
            run_visualization(cmd_args)
