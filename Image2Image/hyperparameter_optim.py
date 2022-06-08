import os
import time
import pytorch_lightning as pl
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from models import Image2ImageModule
from helper import SEP

SAVE_PATH = SEP.join(['Image2Image', 'Optimization'])
if not os.path.isdir: os.mkdir(SAVE_PATH)
MODEL_FILENAME = 'best_optuna_model_2.ckpt'

def hyperparameter_optimization(mode: str, datamodule: pl.LightningDataModule, n_trials: int, epochs_per_trial: int, cuda_device, limit_train_batches = None, limit_val_batches = None):

    assert mode in ['grayscale', 'rgb', 'bool', 'segmentation'], 'Unknown mode setting!'

    def objective(trial: optuna.trial.Trial) -> float:

        print(f'\nSTARTING NEW TRIAL {trial.number+1}/{n_trials} WITH {epochs_per_trial} EPOCHS PER TRIAL...\n\n')

        # Hyperparameters to be optimized
        lr_scheduler = trial.suggest_categorical("sch", ['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau'])
        lr_sch_step_size = trial.suggest_int("step_size", 5, 12)
        lr_sch_gamma = trial.suggest_float("gamma", 0.05, 0.2)
        learning_rate = trial.suggest_float("lr", 5e-4, 5e-3)

        module = Image2ImageModule(mode=mode, learning_rate=learning_rate, lr_scheduler=lr_scheduler)

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

        hyperparameters = dict(lr=learning_rate, sch=lr_scheduler)
        trainer.logger.log_hyperparams(hyperparameters)
        start_training_time = time.time()
        trainer.fit(module, datamodule=datamodule)

        print(f'\n\nTRIAL FINISHED AFTER {(time.time()-start_training_time)/60.:.3f} minutes with val_loss = {trainer.callback_metrics["val_loss"].item():.3f}\n')

        trial.set_user_attr(key="best_model", value = module.net.state_dict())

        return trainer.callback_metrics["val_loss"].item()

    def callback(study: optuna.study.Study, trial: optuna.trial.Trial):
        if study.best_trial.number == trial.number:
            print(f'Best value improved to {study.best_trial.value:.3f} in trial id {study.best_trial.number}\n')
            study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params in best trial: ")
    for key, value in best_trial.params.items():
        if key != "best_model":
            print("    {}: {}".format(key, value))

    # save best model_dict
    if 'best_model' in study.user_attrs:
        print("\nSaving best model from hyperparameter optimization...")
        torch.save(study.user_attrs['best_model'], os.path.join(SAVE_PATH, MODEL_FILENAME))
    
    fig = optuna.visualization.plot_slice(study, params=['lr', 'sch'])
    fig.write_image(os.path.join(SAVE_PATH, 'params.png'))

    return best_trial