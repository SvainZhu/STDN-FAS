import os
import torch


# Base Configuration Class
class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # GPU Usage
    GPU_USAGE = '0'
    DATABASE = 'OULU'
    PROTOCOL = '_1'
    CROP_SIZE = '1.6'
    INTERVAL = '6'

    # Log and Model Storage Default
    LOG_DIR = '../log/%s/%s/' % (DATABASE, CROP_SIZE)
    LOG_DEVICE_PLACEMENT = False

    # Input Data Meta
    IMAGE_SIZE = 256
    MAP_SIZE = 32

    # Training Meta
    BATCH_SIZE = 5
    G_D_RATIO = 2
    LEARNING_RATE = 5e-5
    LEARNING_MOMENTUM = 0.999
    MAX_EPOCH = 30
    NUM_EPOCHS_PER_DECAY = 3
    WEIGHT_AVERAGE_DECAY = 5e-5
    GAMMA = 0.3

    def __init__(self, gpu, database, protocol):
        """Set values of computed attributes."""
        self.DATABASE = database
        self.PROTOCOL = protocol
        self.GPU_USAGE = gpu
        self.compile()

    def compile(self):
        if not os.path.isdir(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        if not os.path.isdir(self.LOG_DIR + '/test'):
            os.makedirs(self.LOG_DIR + '/test')
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and a[0].isupper():
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
