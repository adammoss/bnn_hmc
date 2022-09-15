# Prebuild datasets in case of issues during training

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bnn_hmc'))

from data_scripts.data_config import config

import tensorflow_datasets as tfds
import astro_datasets


for c in config:
    ds, info = tfds.load(name=c['dataset'], split=c['train'], with_info=True, as_supervised=True,
                         builder_kwargs=c['builder_kwargs'])
    ds, info = tfds.load(name=c['dataset'], split=c['test'], with_info=True, as_supervised=True,
                         builder_kwargs=c['builder_kwargs'])
    ds, info = tfds.load(name=c['dataset'], split=c['eval'], with_info=True, as_supervised=True,
                         builder_kwargs=c['builder_kwargs'])
    for ood in c['ood']:
        ds, info = tfds.load(name=ood['dataset'], split=ood['eval'], with_info=True, as_supervised=True,
                             builder_kwargs=ood['builder_kwargs'])
