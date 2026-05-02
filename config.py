config = {
    'max_epoch' : 50,
    'log_train' : 100,
    'lr' : 1e-3,
    'starting_epoch' : 0,
    'batch_size' : 8,
    'log_val' : 10,
    'task' : 'acl', # "meniscus" and  "acl" are the other options
    'weight_decay' : 1e-4,
    'momentum' : 0.9,
    'patience' : 5,
    'save_model' : 1,
    'exp_name' : 'test',
    # Colab-friendly defaults to reduce GPU memory
    'image_size' : 224,
    'target_slices' : 24,
    'num_workers' : 4,
    'num_runs' : 5,
    'base_seed' : 42,
    'use_amp' : True,
    'use_data_parallel' : True,
    'gpu_ids' : [0, 1],
    # Warm-start from abnormal best checkpoint (optional)
    'use_warm_start' : False,
    'warm_start_path' : 'weights/abnormal/run_01/efficientnetb0_sa_best_model.pth',
    'warm_start_tasks' : ['acl', 'meniscus'], # use ['all'] to apply all tasks
    'warm_start_strict' : True,
}
