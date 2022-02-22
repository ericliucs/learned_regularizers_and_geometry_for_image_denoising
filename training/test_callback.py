from tensorflow.keras.callbacks import Callback
import numpy as np
from typing import Dict
from generator.datagenerator import DataGenerator
def psnr(x, y):
    return 20*np.log10(1.0/np.sqrt(np.mean((x-y) ** 2)))

class PSNRSSIMTest(Callback):
    def __init__(self, config: Dict, test_every: int):
        super(PSNRSSIMTest, self).__init__()
        self.config = config
        self.test_every = test_every
        self.data_generator = DataGenerator(config, load_training_data=False)
        super(PSNRSSIMTest, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.test_every == 0:
            psnrs = []
            #ssims = []
            for noisy, clean in self.data_generator.generate_test_set():
                denoised = self.model.predict(noisy)
                psnrs.append(psnr(denoised, clean))

            psnr_avg = np.mean(psnrs)
           ##ssim_avg = np.mean(ssims)

            #print(f'{self.config["test"]}: PSNR = {psnr_avg:2.2f}dB, SSIM = {ssim_avg:1.4f}')
            print(f'{self.config["test"]}: PSNR = {psnr_avg:2.2f}dB')
