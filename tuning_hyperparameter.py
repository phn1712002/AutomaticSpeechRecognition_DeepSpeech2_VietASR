# Environment Variables
PATH_CONFIG = './tuning_hyperparameter.json'
PATH_DATASET = './Dataset/'
PATH_LOGS = './Checkpoint/logs/'
PATH_TENSORBOARD = './Checkpoint/tensorboard/'
PATH_TFLITE = './Checkpoint/export/'

# Argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_config', type=bool, default=False, help='Pretrain model DeepSpeech2 in logs training in dataset')
parser.add_argument('--path_file_pretrain', type=str, default='', help='Path file pretrain model')
parser.add_argument('--export_tflite', type=bool, default=False, help='Export to tflite')
args = parser.parse_args()

# Get config
from Tools.Json import loadJson
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_wandb', 'config_sweep', 'config_other', 'config_dataset']
    if all(key in config for key in keys_to_check):
        config_wandb = config['config_wandb']
        config_sweep = config['config_sweep']
        config_other = config['config_other']
        config_dataset = config['config_dataset']
    else:
        raise RuntimeError('Error config')

# Turn off warning
import warnings
if not config_other['warning']:
    warnings.filterwarnings('ignore')

# Load vocabulary
from Tools.NLP import Tokenization
list_vocabulary = loadJson(path=config_dataset['path_vocab'])
vocabulary = Tokenization().settingWithList(list_vocabulary['vocabulary'])

# Load dataset
from Dataset.Createdataset import DatasetDeepSpeech2
train_dataset_raw, dev_dataset_raw, test_dataset_raw = DatasetDeepSpeech2(path=PATH_DATASET)()

# Tuning Hyperparamter
import wandb
def tuningHyperparameter(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Create config
        config_model = {
            
        }
        
        config_opt = {
            
        }
        
        config_train = {
            
        }
        config_dataset = {
            
        }
        
        # Create pipeline
        from Architecture.Pipeline import PipelineDeepSpeech2
        pipeline = PipelineDeepSpeech2(vocabulary=vocabulary, config_model=config_model)

        train_dataset = PipelineDeepSpeech2(vocabulary=vocabulary, 
                                        augmentation=config_dataset['augmentation'], 
                                        params_noise=config_dataset['params_noise'], 
                                        config_model=config_model)(dataset=train_dataset_raw, batch_size=config_dataset['batch_size_train'])

        dev_dataset = PipelineDeepSpeech2(vocabulary=vocabulary,   
                                        config_model=config_model)(dataset=dev_dataset_raw, batch_size=config_dataset['batch_size_dev'])


        # Create optimizers
        from Optimizers.OptimizersDeepSpeech2 import CustomOptimizers
        opt_dp2 = CustomOptimizers(**config_opt)()

        # Callbacks 
        from Tools.Callbacks import CreateCallbacks
        callbacks_DP2 = CreateCallbacks(PATH_TENSORBOARD=PATH_TENSORBOARD, 
                                        PATH_LOGS=PATH_LOGS, 
                                        config=config, 
                                        train_dataset=train_dataset, 
                                        dev_dataset=dev_dataset, 
                                        pipeline=pipeline)
            
        # Create model
        from Architecture.Model import DeepSpeech2
        dp2 = DeepSpeech2(vocabulary=vocabulary, 
                        opt=opt_dp2,
                        **config_model).build(summary=config_other['summary'])

        # Pretrain
        from Tools.Weights import loadNearest, loadWeights
        if args.pretrain_config:
            if args.path_file_pretrain == '':
                dp2 = loadNearest(class_model=dp2, path_folder_logs=PATH_LOGS)
            else: 
                dp2 = loadWeights(class_model=dp2, path=args.path_file_pretrain)

        # Train model
        dp2.fit(train_dataset=train_dataset, 
                dev_dataset=dev_dataset, 
                callbacks=callbacks_DP2,
                epochs=config_train['epochs'])