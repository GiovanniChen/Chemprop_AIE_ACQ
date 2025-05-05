import chemprop

def main():
    arguments = [
        # '--seed', '75',
        # # 拆分数据集使用的随机种子
        # '--pytorch_seed', '0',

        # '--data_path', 'AIEACQ_train.csv',
        '--dataset_type', 'classification',
        # 'regression', 'classification', 'multiclass', 'spectra'
        # '--multiclass_num_classes', '4',
        '--num_workers', '4',
        # 并行加载工作线程数
        '--number_of_molecules', '1',
        '--split_sizes', '0.9', '0.1', '0',
        '--split_type', 'random',
        # 'random', 'scaffold_balanced'
        # 'predetermined', 'crossval'
        # 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'
        '--separate_test_path', 'AIEACQ_test.csv',

        '--ensemble_size', '5',

        '--features_generator', 'rdkit_2d_normalized',
        # 'morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized'
        '--no_features_scaling',
        # When using rdkit_2d_normalized features, --no_features_scaling must be specified.
        
        '--depth', '5',
        '--hidden_size', '1300',
        '--ffn_num_layers', '2',
        '--ffn_hidden_size', '1400',
        '--init_lr', '0.002',
        '--max_lr', '0.003',
        '--final_lr', '0.0002',
        '--dropout', '0.25',

        '--activation', 'ReLU',
        # 激活函数，默认ReLU
        # 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'
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
        '--warmup_epochs', '4',
        # '--num_folds', '10',

        '--save_dir', 'checkpoints_x'
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

if __name__ == '__main__':
    main()
