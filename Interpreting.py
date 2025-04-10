import chemprop

def main():
    arguments = [
        # '--num_workers', '0',
        # 并行加载工作线程数
        '--data_path', './data/AIE_ACQ/AIEACQ_interpret.csv',
        '--features_generator', 'rdkit_2d_normalized',
        # 'morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized'
        '--no_features_scaling',
        # When using rdkit_2d_normalized features, --no_features_scaling must be specified.
        '--checkpoint_dir', '20240503_checkpoints_RDKit_best_forinterpret',

        '--max_atoms', '50',
        '--min_atoms', '15',

        '--batch_size', '500',
        '--rollout', '20',

        '--prop_delta', '0.5',
        '--property_id', '1'
    ]
    args = chemprop.args.InterpretArgs().parse_args(arguments)
    chemprop.interpret.interpret(args=args)

if __name__ == '__main__':
    main()