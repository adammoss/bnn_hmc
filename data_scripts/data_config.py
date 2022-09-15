config = [
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
        'ood': [{
            'dataset': 'mirabest/uncertain',
            'eval': 'test',
            'builder_kwargs': None,
        }]
    },
    {
        'dataset': 'slc/space',
        'train': 'train[:80%]',
        'test': 'train[80%:]',
        'eval': 'test',
        'batch_size': 50,
        'ensemble_epochs': 20,
        'subset_train_to': None,
        'scaling': None,
        'builder_kwargs': None,
        'ood': []
    },
    {
        'dataset': 'mlsst/Y10',
        'train': 'train',
        'test': 'validation',
        'eval': 'test',
        'batch_size': 50,
        'ensemble_epochs': 100,
        'subset_train_to': 20000,
        'scaling': 'asinh',
        'builder_kwargs': None,
        'ood': [{
            'dataset': 'mlsst/Y1',
            'eval': 'test',
            'builder_kwargs': None,
        }]
    },
    {
        'dataset': 'cmd',
        'train': 'train[:90%]',
        'test': 'train[90%:95%]',
        'eval': 'train[95%:]',
        'batch_size': 50,
        'ensemble_epochs': 100,
        'subset_train_to': None,
        'scaling': 'asinh',
        'builder_kwargs': {"simulation": "IllustrisTNG", "field": "Mtot", "parameters": ["omegam"]},
        'ood': [{
            'dataset': 'cmd',
            'eval': 'train[95%:]',
            'builder_kwargs': {"simulation": "SIMBA", "field": "Mtot", "parameters": ["omegam"]},
        }]
    },
]
