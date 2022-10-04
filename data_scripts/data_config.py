model = 'lenet'
num_repeats = 3
num_ensemble_repeats = 5
image_size = 64
subset_train_to = 5000

config = [
    {
        'dataset': 'cmd',
        'train': 'train[:90%]',
        'test': 'train[90%:95%]',
        'eval': 'train[95%:]',
        'batch_size': 100,
        'scaling': 'asinh',
        'builder_kwargs': '{"simulation": "IllustrisTNG", "field": "Mtot", "parameters": ["omegam"]}',
        'ood': [{
            'dataset': 'cmd',
            'eval': 'train[95%:]',
            'builder_kwargs': '{"simulation": "SIMBA", "field": "Mtot", "parameters": ["omegam"]}',
        }],
        'optimizer': 'SGD',
        'step_size': 1e-7,
    },
    {
        'dataset': 'mirabest/confident',
        'train': 'train[:80%]',
        'test': 'train[80%:]',
        'eval': 'test',
        'batch_size': 53,
        'scaling': None,
        'builder_kwargs': None,
        'ood': [{
            'dataset': 'mirabest/uncertain',
            'eval': 'test',
            'builder_kwargs': None,
        }],
        'optimizer': 'SGD',
        'step_size': 3e-7,
    },
    {
        'dataset': 'slc/space',
        'train': 'train[:80%]',
        'test': 'train[80%:]',
        'eval': 'test',
        'batch_size': 100,
        'ensemble_epochs': 20,
        'scaling': None,
        'builder_kwargs': None,
        'ood': [],
        'optimizer': 'SGD',
        'step_size': 3e-7,
    },
    {
        'dataset': 'mlsst/Y10',
        'train': 'train',
        'test': 'validation',
        'eval': 'test',
        'batch_size': 100,
        'scaling': 'asinh',
        'builder_kwargs': None,
        'ood': [{
            'dataset': 'mlsst/Y1',
            'eval': 'test',
            'builder_kwargs': None,
        }],
        'optimizer': 'Adam',
        'step_size': 1e-5,
    },
]
