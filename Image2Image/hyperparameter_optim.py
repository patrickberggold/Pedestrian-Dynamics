import pytorch_lightning as pl
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from models import Image2ImageModule

TRIAL_EPOCHS = 12

def hyperparameter_optimization(mode: str, datamodule: pl.LightningDataModule, n_trials: int, cuda_device):

    assert mode in ['grayscale', 'rgb', 'bool', 'segmentation'], 'Unknown mode setting!'
    
    def objective(trial: optuna.trial.Trial) -> float:

        print(f'\nSTARTING NEW TRIAL {trial.number+1}/{n_trials}...\n\n')

        # n_layers = trial.suggest_int("n_layers", 1, 3)
        # output_dims = [
        #     trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
        # ]
        learning_rate = trial.suggest_float("lr", 5e-4, 1e-2)

        module = Image2ImageModule(mode=mode, learning_rate=learning_rate)

        trainer = pl.Trainer(
            logger=True,
            checkpoint_callback=False,
            max_epochs=TRIAL_EPOCHS,
            gpus = [cuda_device], 
            devices=f'cuda:{str(cuda_device)}',
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            # limit_train_batches=2,
            # limit_val_batches=2,
        )

        hyperparameters = dict(lr=learning_rate)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(module, datamodule=datamodule)

        trial.set_user_attr(key="best_model", value = module.net.state_dict())

        return trainer.callback_metrics["val_loss"].item()

    def callback(study: optuna.study.Study, trial: optuna.trial.Trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # optuna.visualization.plot_optimization_history(study)
    # optuna.visualization.plot_slice(study, params=['x', 'y'])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        if key != "best_model":
            print("    {}: {}".format(key, value))
    
    return best_trial