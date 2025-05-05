import chemprop

def main():
    arguments = [
        '--data_path', 'AIEACQ_train.csv',
        '--dataset_type', 'classification',
        # 'regression', 'classification', 'multiclass', 'spectra'
        # '--multiclass_num_classes', '4',
        '--num_workers', '4',
        # 并行加载工作线程数
        '--number_of_molecules', '1',
        '--split_sizes', '0.9', '0.1', '0',
        '--split_type', 'cv-no-test',
        # 'random', 'scaffold_balanced'
        # 'predetermined', 'crossval'
        # 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'
        '--separate_test_path', 'AIEACQ_test.csv',

        '--features_generator', 'rdkit_2d_normalized',
        # 'morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized'
        '--no_features_scaling',
        # When using rdkit_2d_normalized features, --no_features_scaling must be specified.

        # '--depth', '3',
        # '--hidden_size', '1400',
        # '--ffn_num_layers', '2',
        # '--ffn_hidden_size', '1400',
        # '--init_lr', '0.0001',
        # '--max_lr', '0.001',
        # '--final_lr', '0.00001',
        # '--dropout', '0.25',

        '--activation', 'ReLU',
        '--loss_function', 'binary_cross_entropy',
        # 'mse', 'bounded_mse'
        # 'binary_cross_entropy', 'cross_entropy'
        # 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'
        '--log_frequency', '10',
        # 每经过多少批次(batch)记录一次训练的损失(loss)，默认10
        '--metric', 'auc',
        '--extra_metric', 'accuracy', 'prc-auc', 'f1', 'mcc',
        # 'auc', 'prc-auc'
        # 'rmse', 'mae', 'mse'
        # 'r2', 'accuracy'
        # 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein'
        # 'f1', 'mcc', 'bounded_rmse', 'bounded_mae', 'bounded_mse'
        # 用于评估模型的额外指标，默认无
        # '--metrics', '',
        # # 评估模型的指标列表，其中第一个指标用于早停判定

        '--batch_size', '100',
        '--epochs', '30',
        '--num_folds', '10',

        '--search_parameter_keywords', 'depth', 'hidden_size', 'ffn_hidden_size', 'ffn_num_layers', 'dropout', 'final_lr', 'init_lr', 'max_lr', 'warmup_epochs',
        # 'basic', 'learning_rate', 'all'
        # activation, aggregation, aggregation_norm, batch_size
        # depth, ffn_hidden_size, ffn_num_layers, dropout, hidden_size,
        # final_lr, init_lr, max_lr, warmup_epochs

        '--num_iters', '100',
        '--config_save_path', 'hyperopt_config.json',
        '--hyperopt_checkpoint_dir', 'hyperopt_checkpoint',
        '--log_dir', 'hyperopt/log',
        '--save_dir', '.hyperopt/hyperopt_save'
    ]

    args = chemprop.args.HyperoptArgs().parse_args(arguments)
    chemprop.hyperparameter_optimization.hyperopt(args=args)

if __name__ == '__main__':
    main()
