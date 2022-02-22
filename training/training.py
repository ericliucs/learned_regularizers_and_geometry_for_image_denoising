from typing import Dict
import os
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from generator.datagenerator import DataGenerator
from training.losses import retrieve_loss_function
from training.test_callback import PSNRSSIMTest


def retrieve_training_function(config: Dict):
    """Retrieves training function by name

    Parameters
    ----------
    config: (Dict) - Configuration that contains training_function key

    Returns
    -------

    """
    if config['training_function'] == 'standard':
        return standard_training
    else:
        raise Exception('training function does not exist')


def standard_training(Model):

    checkpointer = ModelCheckpoint(os.path.join(Model.model_dir, 'model_{epoch:04d}'),
                                   verbose=1, save_weights_only=False, period=10)
    loss_file = os.path.join(Model.model_dir, 'loss_log.csv')
    csv_logger = CSVLogger(loss_file, append=True, separator=',')
    lr_scheduler = LearningRateScheduler(Model.scheduler())
    tester = PSNRSSIMTest(Model.config, test_every=2)

    Model.model.compile(optimizer=Model.optimizer(),  loss=retrieve_loss_function(Model.config))

    data_generator = DataGenerator(Model.config, load_training_data=True)
    Model.model.fit(data_generator,
                            epochs=Model.config['epochs'], verbose=1,
                            callbacks=[lr_scheduler, checkpointer, csv_logger,tester],
                            initial_epoch=Model.epochs)



