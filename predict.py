import chemprop

def main():
    arguments = [
        '--test_path', './data/AIE_ACQ/data.csv',
        '--preds_path', '¡£/data/AIE_ACQ/data_preds.csv',
        '--checkpoint_dir', '20240503_checkpoints_RDKit_best',
        # '--checkpoint_dir', '20240503_checkpoints_RDKit_best_forinterpret',
        '--num_workers', '4',

        '--features_generator', 'rdkit_2d_normalized',
        # 'morgan', 'morgan_count', 'rdkit_2d', 'rdkit_2d_normalized'
        '--no_features_scaling'
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    chemprop.train.make_predictions(args=args)

if __name__ == '__main__':
    main()