import numpy as np
import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, test_data, save_model_path, batch_size=16, fold=None):
        super(CustomCallback, self).__init__()
        self.valid_inputs = valid_data[0]
        self.valid_labels = valid_data[1]
        self.test_inputs = test_data
        self.save_model_path = save_model_path
        self.batch_size = batch_size
        self.fold = fold
        self.valid_predictions = []
        self.test_predictions = []

    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(self.model.predict(x=self.valid_inputs, batch_size=self.batch_size))
        rho_val = self._compute_spearmanr(self.valid_labels, np.average(self.valid_predictions, axis=0))

        print(f"\nvalidation rho: {round(rho_val, 4)}")

        if self.fold is not None:
            model_name = self.save_model_path.split('/')[-2]
            self.model.save_weights(self.save_model_path + f'{model_name}-base-{self.fold}-{epoch}-{rho_val}.h5')

        self.test_predictions.append(self.model.predict(x=self.test_inputs, batch_size=self.batch_size))

    def _compute_spearmanr(self, trues, prds):
        from scipy.stats import spearmanr
        rhos = []
        for col_trues, col_pred in zip(trues.T, prds.T):
            rhos.append(spearmanr(col_trues, col_pred).correlation)
        return np.nanmean(rhos)
