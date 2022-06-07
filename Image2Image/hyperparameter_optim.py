import pytorch_lightning as pl
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from models import Image2ImageModule

PERCENT_VALID_EXAMPLES = 0.1
EPOCHS = 10

def hyperparameter_optimization(mode: str, datamodule: pl.LightningDataModule, n_trials: int, cuda_device):

    assert mode in ['grayscale', 'rgb', 'bool', 'segmentation'], 'Unknown mode setting!'
    
    def objective(trial: optuna.trial.Trial) -> float:

        # n_layers = trial.suggest_int("n_layers", 1, 3)
        # dropout = trial.suggest_float("dropout", 0.2, 0.5)
        # output_dims = [
        #     trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
        # ]
        learning_rate = trial.suggest_float("lr", 1e-2, 5e-4)

        module = Image2ImageModule(mode=mode, learning_rate=learning_rate)

        trainer = pl.Trainer(
            logger=True,
            limit_val_batches=PERCENT_VALID_EXAMPLES,
            checkpoint_callback=False,
            max_epochs=EPOCHS,
            gpus = [cuda_device], 
            devices=f'cuda:{str(cuda_device)}',
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        )
        # hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
        hyperparameters = dict(lr=learning_rate)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(module, datamodule=datamodule)

        return trainer.callback_metrics["val_acc"].item()

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)

    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
    
    return best_trial