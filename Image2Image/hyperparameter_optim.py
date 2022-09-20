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
    mode: str, 
    datamodule: pl.LightningDataModule, 
    n_trials: int, 
    epochs_per_trial: int, 
    folder_name: str,
    cuda_device: int = 0,
    test_run: bool = True,
    ):

    if test_run: n_trials = 2
    if test_run: epochs_per_trial = 2
    limit_train_batches = 2 if test_run else None
    limit_val_batches = 2 if test_run else None

    assert mode in ['grayscale', 'grayscale_movie', 'evac'], 'Unknown mode setting!'
    
    ROOT_PATH = SEP.join(['Image2Image', 'Optimization'])
    folder_path = os.path.join(ROOT_PATH, folder_name)
    if not os.path.isdir(folder_path): os.mkdir(folder_path)

    MODEL_FILENAME = 'best_model.ckpt'
    LOG_FILENAME = os.path.join(folder_path, 'log.txt')

    log_file = open(LOG_FILENAME, "w")
    log_file.close()

    def objective(trial: optuna.trial.Trial) -> float:

        # Hyperparameters to be optimized
        learning_rate = trial.suggest_float("lr", 8e-5, 8e-4)
        lr_scheduler = trial.suggest_categorical("sch", ['ReduceLROnPlateau', 'ExponentialLR'])
        dropout = trial.suggest_float("dropout", 0.02, 0.45)
        # opt = trial.suggest_categorical("opt", ['Adam', 'AdamW'])
        gamma = trial.suggest_float('gamma', 0.1, 0.3)
        # batch_size = trial.suggest_categorical("batch_size", [4, 8])
        # weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3])

        # datamodule.set_batch_size(batch_size)

        module = Image2ImageModule(mode=mode, unfreeze_backbone_at_epoch=None, lr_scheduler=lr_scheduler, learning_rate=learning_rate, alternate_unfreezing=True, lr_sch_gamma=gamma, weight_decay=1e-6, p_dropout=dropout)

        # Load from checkpoint
        CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'trained_img_and_evac.ckpt'])
        state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH).items()])

        module.load_state_dict(OrderedDict((k_module, v_loaded) for (k_loaded, v_loaded), (k_module, v_module) in zip(state_dict.items(), module.state_dict().items())))

        # TODO implement feedback when trials are pruned
        hyperparameters = dict(lr=learning_rate, sch=lr_scheduler, gamma=gamma, dropout=dropout)

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