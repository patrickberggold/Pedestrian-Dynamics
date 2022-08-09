from src.utils import timed_main, set_seed
from src.parser import main_parser
from src.data_pre_process import Trajectory_Data_Pre_Process
from src.trainer import trainer

conf = {
    'train': {
        'SAR': {
            'model_name': 'SAR',
            'phase': 'train',
            'num_epochs': 500,
            'batch_size': 32,
            'validate_every': 1,
            'learning_rate': 0.001
        },
        'Goal_SAR': {
            'model_name': 'Goal_SAR',
            'phase': 'train',
            'num_epochs': 300,
            'batch_size': 32,
            'validate_every': 1,
            'learning_rate': 0.0001
        }
    },

    'test': {
        'SAR': {
            'model_name': 'SAR',
            'phase': 'test',
            'load_checkpoint': 'best',
            'batch_size': 32
        },
        'Goal_SAR': {
            'model_name': 'Goal_SAR',
            'phase': 'test',
            'load_checkpoint': 'best',
            'batch_size': 32
        }
    }
}

@timed_main(use_git=True)
def main():
    MODE = 'train'
    ARCH = 'Goal_SAR'
    CUDA_DEVICE = 1
    conf['device'] = CUDA_DEVICE
    conf['dataset'] = 'ind'
    # Parse input parameters and save/load config file
    assert MODE in ['train', 'test'] and ARCH in ['SAR', 'Goal_SAR']
    args = main_parser(MODE, ARCH, conf)

    # Set seed for reproducibility
    if args.reproducibility:
        set_seed(seed_value=12345, use_cuda=args.use_cuda)

    # Run data pre-process
    Trajectory_Data_Pre_Process(args)

    if args.phase == 'pre-process':
        processor = None
    else:
        # Initialize data-loader and model
        processor = trainer(args, full_dataset=False)

    if args.phase == 'pre-process':
        print("Data pre-processing and batches creation finished.")
    elif args.phase == 'train':
        processor.train()
    elif args.phase == 'test':
        processor.test(load_checkpoint=args.load_checkpoint)
    elif args.phase == 'train_test':
        processor.train_test()
    else:
        raise ValueError(
            f"Unsupported phase {args.phase}! args.phase can only take the "
            f"following values: 'train', 'test', 'train_test' or 'pre-process'")


if __name__ == '__main__':
    main()
