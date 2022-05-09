import csv
from tensorflow.keras.callbacks import Callback
from generator.datagenerator import DataGenerator
import os


class PSNRTest(Callback):
    """Callback to get average PSNR on test set during training"""

    def __init__(self, cnn_model, test_every: int):
        """Initialize the callback with CNN model

        Parameters
        ----------
        cnn_model: (CNNModel) - CNNModel Object to be used to get model configuration and save locations
        test_every: (int) - Tests the model every test_every epochs
        """
        super(PSNRTest, self).__init__()
        self.cnn_model = cnn_model
        self.test_every = test_every
        self.data_generator = DataGenerator(self.cnn_model.config, load_training_data=False)
        self.output_file = os.path.join(self.cnn_model.model_dir, 'psnr_test.csv')
        super(PSNRTest, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        """Tests the model on epoch end"""

        # Initialize csv file
        if epoch == 0:
            with open(self.output_file, "w") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['epoch', f'rmse on {self.cnn_model.config["test"]}',
                                 f'psnr on {self.cnn_model.config["test"]}'])

        # Test the model
        if (epoch+1) % self.test_every == 0:

            metrics = self.cnn_model.test()
            print(f'{self.cnn_model.config["test"]}: PSNR = {metrics["PSNR"]:2.2f}dB')

            output_file = os.path.join(self.cnn_model.model_dir, 'psnr_test.csv')
            with open(output_file, "a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([epoch, metrics['RMSE'], metrics['PSNR']])
