class Config:
    # Dataset
    dataset_name = 'CICIDS2017'
    data_root = './data'
    batch_size = 128
    num_workers = 4
    z_dim = 128
    gan_hidden_dim = 512
    input_dim = 80  # Adjust based on your dataset
    num_classes = 7  # Adjust based on your dataset
    clf_epochs = 200
    gan_epochs = 300
    clf_lr = 1e-3
    gan_lr = 2e-4
    clf_lr_patience = 5
    max_grad_norm = 1.0

    # Augmentation
    augmentation_target_mode = "second_max"
    max_synthetic_multiplier = 1.5
    max_fallback_rate = 0.05

    # Classification Loss
    clf_class_weighting = "none"
    clf_effective_num_beta = 0.9999
    clf_label_smoothing = 0.1
    hidden_loss_weight = 1.0