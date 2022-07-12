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
 
ROOT_PATH = SEP.join(['Image2Image', 'Optimization'])
OPTIMIZATION_NAME = 'grayscale_thickness_5_gammaStep_5_plusCosAnn_RedPlat_contFromTrain'
FOLDER_NAME = os.path.join(ROOT_PATH, OPTIMIZATION_NAME)
if not os.path.isdir(FOLDER_NAME): os.mkdir(FOLDER_NAME)

MODEL_FILENAME = 'model_optuna_grayscale_thickness_5_gammaStep_5.ckpt'
LOG_FILENAME = os.path.join(FOLDER_NAME, 'log.txt')
# TODO implement visual TF as backbone
def hyperparameter_optimization(
    mode: str, 
    datamodule: pl.LightningDataModule, 
    n_trials: int, 
    epochs_per_trial: int, 
    cuda_device: int = 0, 
    limit_train_batches = None, 
    limit_val_batches = None
    ):

    assert mode in ['grayscale', 'rgb', 'bool', 'segmentation', 'timeAndId', 'grayscale_movie'], 'Unknown mode setting!'

    log_file = open(LOG_FILENAME, "w")
    log_file.close()

    def objective(trial: optuna.trial.Trial) -> float:

        # Hyperparameters to be optimized
        learning_rate = trial.suggest_float("lr", 5e-4, 1e-2)
        lr_scheduler = trial.suggest_categorical("sch", ['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau'])
        # lr_sch_step_size = trial.suggest_int("step_size", 4, 12)
        lr_sch_gamma = trial.suggest_float("gamma", 0.3, 0.6)
        # non_traj_vals = trial.suggest_float("ntv", -7., -0.5)
        # unfreeze_backbone_epoch = trial.suggest_categorical
        # max_traj_val = trial.suggest_float("mtv", )

        # CKPT_PATH = SEP.join(['Image2Image', 'checkpoints', 'checkpoints_DeepLab4Img2Img', 'model_grayscale_lineThickness5_CosAnn_Step5_Lr122_Gam42_epoch=36-step=5772.ckpt'])
        # state_dict = OrderedDict([(key.replace('net.', ''), tensor) if key.startswith('net.') else (key, tensor) for key, tensor in torch.load(CKPT_PATH)['state_dict'].items()])
        module = Image2ImageModule(mode=mode, learning_rate=learning_rate, lr_scheduler=lr_scheduler, lr_sch_step_size=5,lr_sch_gamma=lr_sch_gamma, relu_at_end = True, num_heads=8)
        # module.net.load_state_dict(state_dict)

        # datamodule = FloorplanDataModule(mode = mode, cuda_index = cuda_device, batch_size = 4, num_ts_per_floorplan=8, vary_area_brightness=False)

        datamodule.set_non_traj_vals(new_val = 0.)

        # TODO implement feedback when trials are pruned
        hyperparameters = dict(lr=learning_rate, gamma=lr_sch_gamma, sch=lr_scheduler)

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

        trial.set_user_attr(key="model_state_dict", value = module.net.state_dict())

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
            jsonFile = open(os.path.join(FOLDER_NAME, 'hyperparameters_'+MODEL_FILENAME.replace('.ckpt', '.json')), 'w')
            jsonFile.write(jsonString)
            jsonFile.close()
            
            # Save best model in ckpt format
            torch.save(trial.user_attrs['model_state_dict'], os.path.join(FOLDER_NAME, MODEL_FILENAME))

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
    fig.write_image(os.path.join(FOLDER_NAME, 'params.png'))

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