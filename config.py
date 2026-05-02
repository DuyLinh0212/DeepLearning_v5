config = {
    'max_epoch' : 50,
    'log_train' : 100,
    'lr' : 1e-4,
    'starting_epoch' : 0,
    'batch_size' : 2,
    'log_val' : 10,
    'task' : 'acl', # "meniscus" and  "acl" are the other options
    'weight_decay' : 1e-4,
    'momentum' : 0.7,
    'patience' : 5,
    'save_model' : 1,
    'exp_name' : 'test',
    # Colab-friendly defaults to reduce GPU memory
    'image_size' : 224,
    'target_slices' : 24,
    'num_workers' : 2,
    'num_runs' : 5,
    'base_seed' : 42,
    'use_amp' : True,
    'use_data_parallel' : True,
    'gpu_ids' : [0, 1],
}
