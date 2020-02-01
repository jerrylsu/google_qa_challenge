import tensorflow as tf


def set_gpu(gpu_id: int):
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('physical devices: {}'.format(len(gpus)))
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print('logical devices: {}'.format(len(logical_gpus)))