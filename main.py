from models.models import TNRD, TotalDeepVariation
import tensorflow as tf

def getGPU():
    """
    Grabs GPU. Sometimes Tensorflow attempts to use CPU when this is not called on my machine.
    From: https://www.tensorflow.org/guide/gpu
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    getGPU()
    TotalDeepVariation({}, verbose=False).summary()
    #TotalDeepVariation({}, verbose=False).summary()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
