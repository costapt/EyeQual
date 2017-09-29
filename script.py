import train
import models
import argparse

from eval.eval import get_auc_score
from keras.callbacks import ModelCheckpoint, EarlyStopping


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='EyeQual')
    """
    Data parameters' definition
    """
    parser.add_argument('--batch_size', type=int, help='The size of the batch to be used.', dest='batch_size', default=8)
    parser.add_argument('--load_data', type=bool, help='If true then loads the data into memory.', dest='load_data', default=True)
    parser.add_argument('--weak_dir', required=True, help='The directory where the data is present.', dest='weak_dir')
    parser.add_argument('--epochs', type=int, default=4000, help='The max number of epochs to train the model.', dest='epochs')

    """
    Dataset augmentation's parameters
    """
    parser.add_argument('--horizontal_flip', type=bool, dest='horizontal_flip', default=False, help='If true, it performs random horizontal flips as a dataset augmentation operation.')
    parser.add_argument('--vertical_flip', type=bool, dest='vertical_flip', default=False, help='If true, it performs random vertical flips as a dataset augmentation operation.')
    parser.add_argument('--width_shift_range', type=float, dest='width_shift_range', default=0.0, help='Horizontal translation, between 0 and 1. Same as in Keras.')
    parser.add_argument('--height_shift_range', type=float, dest='height_shift_range', default=0.0, help='Vertical translation, between 0 and 1. Same as in Keras.')
    parser.add_argument('--rotation_range', type=int, dest='rotation_range', default=0, help='Rotation in degrees. Same as in Keras.')
    parser.add_argument('--zoom_range', type=float, dest='zoom_range', default=0.0, help='Scale in percentage, between 0 and 1. Same as in Keras.')

    """
    Model parameter's definition
    """
    parser.add_argument('--nf', type=int, dest='nf', default=64, help='Number of filters of first layer.')
    parser.add_argument('--n_blocks', type=int, dest='n_blocks', default=4, help='Depth of the network.')
    parser.add_argument('--input_size', type=int, dest='input_size', default=512, help='Size of the input images.')
    parser.add_argument('--pooling_wreg', type=float, dest='pooling_wreg', default=0, help='Regularization term for the weights of the SWAP layer.')
    parser.add_argument('--pooling_breg', type=float, dest='pooling_breg', default=0, help='Regularization term for the bias of the SWAP layer.')
    parser.add_argument('--lr', type=float, dest='lr', default=2e-4, help='Learning rate.')

    def pooling_type(value):
        if value in ['SWAP', 'WAP', 'AP', 'MP']:
            return value
        raise argparse.ArgumentTypeError("{0} is an invalid pooling operation. Must be one of SWAP, WAP, AP or MP.".format(value))

    parser.add_argument('--pooling', type=pooling_type, dest='pooling', default='SWAP', help='One of SWAP, WAP, AP or MP.')

    """
    Callbacks' definition
    """
    parser.add_argument('--experiment_path', type=str, dest='experiment_path', default='swap.hdf5', help='The path where to save the best model.')
    parser.add_argument('--patience', type=int, dest='patience', default=0, help='The early stopping patience. If <= 0, then do not use early stopping.')

    args = parser.parse_args()

    checkpointer = ModelCheckpoint(filepath=args.experiment_path, verbose=1,
                                   save_best_only=True, save_weights_only=False)
    callbacks = [checkpointer]
    if args.patience > 0:
        early = EarlyStopping(patience=args.patience, verbose=1)
        callbacks.append(early)

    aug_params = {'horizontal_flip': args.horizontal_flip,
                  'vertical_flip': args.vertical_flip,
                  'width_shift_range': args.width_shift_range,
                  'height_shift_range': args.height_shift_range,
                  'rotation_range': args.rotation_range,
                  'zoom_range': args.zoom_range}

    """
    Load data
    """
    train_it, val_it, test_it = train.get_data_iterators(batch_size=args.batch_size, data_dir=args.weak_dir,
                                                         target_size=(args.input_size, args.input_size), rescale=1/255.,
                                                         fill_mode='constant', load_train_data=args.load_data,
                                                         color_mode='rgb', **aug_params)

    """
    Train
    """
    eyequal, heatmap = models.quality_assessment(args.nf, input_size=args.input_size, n_blocks=args.n_blocks, lr=args.lr,
                                                 pooling_wreg=args.pooling_wreg, pooling_breg=args.pooling_breg)
    eyequal.fit_generator(train_it, train_it.n, args.epochs, validation_data=val_it, nb_val_samples=val_it.n,
                          verbose=2, callbacks=callbacks)

    """
    Evaluate
    """
    eyequal.load_weights(args.experiment_path)
    print 'Train AUC = {0}'.format(get_auc_score(eyequal, train_it, train_it.n))
    print 'Validation AUC = {0}'.format(get_auc_score(eyequal, val_it, val_it.n))
    print 'Test AUC = {0}'.format(get_auc_score(eyequal, test_it, test_it.n))
