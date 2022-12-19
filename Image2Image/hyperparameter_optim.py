import os
import time
import pytorch_lightning as pl
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from models import Image2ImageModule
from Datamodules import FloorplanDataModule
from helper import SEP
import json
from collections import OrderedDict
 
def hyperparameter_optimization(
    config: dict,
    train_config: str,
    datamodule: pl.LightningDataModule, 
    n_trials: int, 
    epochs_per_trial: int, 
    folder_name: str,
    cuda_device: int = 0,
    ):

    mode = config['mode']
    test_run = config['run_test_epoch']
    if test_run: n_trials = 2
    if test_run: epochs_per_trial = 2
    limit_train_batches = 2 if test_run else None
    limit_val_batches = 2 if test_run else None

    assert mode in ['grayscale', 'grayscale_movie', 'evac', 'evac_only'], 'Unknown mode setting!'
    
    ROOT_PATH = SEP.join(['Image2Image', 'Optimization'])
    folder_path = os.path.join(ROOT_PATH, folder_name)
    if not os.path.isdir(folder_path): os.mkdir(folder_path)

    MODEL_FILENAME = 'best_model.ckpt'
    LOG_FILENAME = os.path.join(folder_path, 'log.txt')

    log_file = open(LOG_FILENAME, "w")
    log_file.close()

    def objective(trial: optuna.trial.Trial) -> float:

        # Hyperparameters to be optimized
        learning_rate = trial.suggest_float("lr", 1e-4, 3e-3)
        # lr_scheduler = trial.suggest_categorical("lr_scheduler", ['ReduceLROnPlateau', 'ExponentialLR', 'CosineAnnealingLR'])
        # gamma = trial.suggest_float('gamma', 0.1, 0.3)
        # batch_size = trial.suggest_categorical("batch_size", [4, 8])
        # weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5])

        # datamodule.set_batch_size(batch_size)

        train_config['learning_rate'] = learning_rate

        module = Image2ImageModule(config=config, train_config=train_config)

        if config['from_ckpt_path']:
            CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', config['from_ckpt_path']])
            state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
            module_state_dict = module.state_dict()

            mkeys_missing_in_loaded = [module_key for module_key in list(module_state_dict.keys()) if module_key not in list(state_dict.keys())]
            lkeys_missing_in_module = [loaded_key for loaded_key in list(state_dict.keys()) if loaded_key not in list(module_state_dict.keys())]

            load_dict = OrderedDict()
            for key, tensor in module_state_dict.items():
                # if (key in state_dict.keys()) and ('decode_head' not in key):
                if key in state_dict.keys():
                    load_dict[key] = state_dict[key]
                else:
                    # if key == 'model.model.classifier.classifier.weight':
                    #     load_dict[key] = state_dict['model.model.classifier.weight']
                    # elif key == 'model.model.classifier.classifier.bias': 
                    #     load_dict[key] = state_dict['model.model.classifier.bias']
                    # else:
                    #     load_dict[key] = tensor
                    load_dict[key] = tensor

            module.load_state_dict(load_dict)

        # TODO implement feedback when trials are pruned
        hyperparameters = dict(lr=learning_rate) # ,lr_scheduler=lr_scheduler

        start_trial_string = f'\n#################################################################################################################' + \
            f'\nSTARTING NEW TRIAL {trial.number+1}/{n_trials} WITH {epochs_per_trial} EPOCHS PER TRIAL with:\n'
        for key in hyperparameters:
            start_trial_string += f'{key}: {hyperparameters[key]}\n'
            trial.set_user_attr(key=key, value = hyperparameters[key])
        print(start_trial_string)
        log_file = open(LOG_FILENAME, "a")
        log_file.write(start_trial_string)
        log_file.close()

        trainer = pl.Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=epochs_per_trial,
            gpus = [cuda_device], 
            devices=f'cuda:{str(cuda_device)}',
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
        )
        trainer.logger.log_hyperparams(hyperparameters)
        start_training_time = time.time()
        trainer.fit(module, datamodule=datamodule)

        finish_trial_string = f'\n\nTRIAL FINISHED AFTER {(time.time()-start_training_time)/60./(module.current_epoch+1):.3f} minutes per epoch with val_loss = {trainer.callback_metrics["val_loss"].item():.3f}\n'
        print(finish_trial_string)
        log_file = open(LOG_FILENAME, "a")
        log_file.write(finish_trial_string)
        log_file.close()

        trial.set_user_attr(key="model_state_dict", value = module.state_dict())

        return trainer.callback_metrics["val_loss"].item()

    def callback(study: optuna.study.Study, trial: optuna.trial.Trial):
        if study.best_trial.number == trial.number:
            json_hyperparams = {}
            string_8 = f'\nBest value improved to {study.best_trial.value:.3f} in trial id {study.best_trial.number+1}\n' + \
                'Saving best model from hyperparameter optimization... Model hyperparameters:\n'
            for key in trial.user_attrs:
                if key != "model_state_dict":
                    key_val = f'{trial.user_attrs[key]:.5f}' if not isinstance(trial.user_attrs[key], str) else f'{trial.user_attrs[key]}'
                    string_8 += f'{key}:{key_val}\n'
                    json_hyperparams.update({key: trial.user_attrs[key]})
            print(string_8)
            
            # write log.txt file
            log_file = open(LOG_FILENAME, "a")
            log_file.write(string_8)
            log_file.close()
            
            # Save hyperparameters in json format
            jsonString = json.dumps(json_hyperparams)
            jsonFile = open(os.path.join(folder_path, 'hyperparameters_'+MODEL_FILENAME.replace('.ckpt', '.json')), 'w')
            jsonFile.write(jsonString)
            jsonFile.close()
            
            # Save best model in ckpt format
            torch.save(trial.user_attrs['model_state_dict'], os.path.join(folder_path, MODEL_FILENAME))

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    string_1 = "\nNumber of finished trials: {}\n".format(len(study.trials))
    print(string_1)
    log_file = open(LOG_FILENAME, "a")
    log_file.write(string_1)

    string_2 = "\nBest trial:\n"
    print(string_2)
    log_file.write(string_2)
    best_trial = study.best_trial

    string_3 = "\n  Value: {}\n".format(best_trial.value)
    print(string_3)
    log_file.write(string_3)

    string_4 = "\n  Params in best trial: \n"
    print(string_4)
    log_file.write(string_4)
    for key, value in best_trial.params.items():
        if key != "model_state_dict":
            string_5 = "\n    {}: {}\n".format(key, value)
            print(string_5)
            log_file.write(string_5)

    fig = optuna.visualization.plot_slice(study, params=[key for key in best_trial.user_attrs if key != 'model_state_dict'])
    fig.write_image(os.path.join(folder_path, 'params.png'))

    string_6 = '\nHyperparameter importances:'
    print(string_6)
    log_file.write(string_6)

    importances = optuna.importance.get_param_importances(study)
    for key in importances:
        string_7 = f'\n{key}: {importances[key]:.3f}'
        print(string_7)
        log_file.write(string_7)

    log_file.close()

    return best_trial