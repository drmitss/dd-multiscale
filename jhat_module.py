# ANN Jacobian module
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
class Jhat:
    # Constructor
    def __init__(self, domdim=2, codomdim=2, layers=[100, 100, 50, 20], batch_size=50, epochs=50, nbr=0, rmax=0.0, learning_rate=0.0001, max_w=0, verbose=True):
        # Store parameters
        self.domdim = domdim
        self.codomdim = codomdim
        self.layers = layers
        self.batch_size = batch_size if batch_size > 0 else 50
        self.epochs = epochs if epochs > 0 else 50
        self.nbr = nbr if nbr >= domdim else 2 * domdim
        self.rmax = rmax
        self.max_norm = max_w
        self.verbose = verbose
        
        # Create and store ANN
        self.model = Sequential()
        
        # Input layer
        # self.model.add(Dense(layers[0], input_shape=(domdim,), activation='swish', kernel_constraint=max_norm(max_w) if max_w > 0 else None))
        self.model.add(Input(shape=(domdim,)))
        self.model.add(Dense(layers[0], activation='swish', kernel_constraint=max_norm(max_w) if max_w > 0 else None))
        
        # Hidden layers
        for n_neurons in layers[1:]:
            self.model.add(Dense(n_neurons, activation='swish', kernel_constraint=max_norm(max_w) if max_w > 0 else None))
        
        # Output layer
        self.model.add(Dense(domdim * codomdim, kernel_constraint=max_norm(100.) if max_w > 0 else None))
        
        self.model.compile(loss=self.create_loss(), optimizer=Adam(learning_rate=learning_rate))
    
    # Internal: loss function generator
    def create_loss(self):
        def loss(real, predict):
            dx = real[:, 0:self.domdim]
            df = real[:, self.domdim:]
            return tf.math.reduce_mean(tf.math.square(tf.math.subtract(df, tf.linalg.matvec(tf.reshape(predict, (self.batch_size, self.codomdim, self.domdim)), dx))))
        return loss
    
    # Internal: prepare the dataset for training
    def prepare_data(self, x, fx, train_mode=True, zero=0):
        if self.verbose:
            print("Preparing data from sample")
            print("Input shape", x.shape)
            print("Output shape", fx.shape)
        
        nbrs = NearestNeighbors(n_neighbors=self.nbr, algorithm='ball_tree').fit(x)
        dist, indices = nbrs.kneighbors(x)
        
        if self.verbose:
            print("Minimal distance:", np.amin(dist))
            print("Average distance:", np.average(dist))
            print("Maximal distance:", np.amax(dist))
        
        if self.rmax == 0:
            self.rmax = np.amax(dist) + 1.0
        
        N = x.shape[0] * self.nbr
        pos = np.empty((N, self.domdim))
        delta = np.empty((N, self.domdim + self.codomdim))
        
        k = 0
        if train_mode:
            for idx in indices[:, 0]:
                for idx2 in indices[idx, 1:]:
                    dx = np.subtract(x[idx2], x[idx])
                    dxnorm = np.linalg.norm(dx)
                    if dxnorm < self.rmax and dxnorm > zero:
                        pos[k] = x[idx]
                        delta[k, 0:self.domdim] = dx / dxnorm
                        delta[k, self.domdim:] = np.subtract(fx[idx2], fx[idx]) / dxnorm
                        k += 1
        else:
            for idx in indices[:, 0]:
                for idx2 in indices[idx, 1:]:
                    dx = np.subtract(x[idx2], x[idx])
                    dxnorm = np.linalg.norm(dx)
                    div = np.linalg.norm(fx[idx2])
                    if dxnorm < self.rmax and dxnorm > 0 and div > zero:
                        pos[k] = x[idx]
                        delta[k, 0:self.domdim] = dx / div
                        delta[k, self.domdim:] = np.subtract(fx[idx2], fx[idx]) / div
                        k += 1
        
        if k < self.batch_size:
            self.batch_size = k
        else:
            r = k % self.batch_size
            if r != 0:
                d = self.batch_size - r
                for j in range(d):
                    pos[k + j] = pos[0]
                    delta[k + j] = delta[0]
                k += d
        
        if self.verbose:
            print("Number of training data points:", k)
            print("Finalized batch size:", self.batch_size)
        
        rand_ind = np.arange(k)
        np.random.shuffle(rand_ind)
        pos = pos[:k]
        delta = delta[:k]
        pos = [pos[j] for j in rand_ind]
        delta = [delta[j] for j in rand_ind]
        
        return (tf.convert_to_tensor(pos[:k]), tf.convert_to_tensor(delta[:k]))
    
    # Fit: train the ANN with a sample set, passed as a parameter.
    # def fit(self, x, fx):
    #     F = self.prepare_data(x, fx)
    #     self.model.fit(F[0], F[1], epochs=self.epochs, batch_size=self.batch_size, verbose=2 if self.verbose else 0)
    #     return self

    # Fit with progress bar
    def fit(self, x, fx):
        # clear any leftover bars
        tqdm._instances.clear()

        F = self.prepare_data(x, fx)

        # build callback list
        callbacks = []
        if self.verbose:
            callbacks.append(TQDMProgressBar())

        # turn off built-in printing
        self.model.fit(
            F[0], F[1],
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=callbacks
        )
        return self

    # Predict: compute the estimate of the Jacobian of F at each entry of x.
    def predict(self, x):
        N = x.shape[0]
        return self.model.predict(x).reshape(N, self.codomdim, self.domdim)
    
    # Predict 1: compute the estimate of the Jacobian of F at a single input x.
    def predict1(self, x):
        return self.model.predict([x])
    
    # Predict flat: same as predict, but does not reshape output
    def predict_flat(self, x):
        return self.model.predict(x)
    
    # Validation functions
    def tangentl(self, j, dx, N):
        return tf.linalg.matvec(tf.reshape(j, (N, self.codomdim, self.domdim)), dx)
    
    def validate(self, x, fx, zero=0):
        F = self.prepare_data(x, fx, train_mode=False, zero=zero)
        dfhat = tf.convert_to_tensor(self.predict(F[0]))
        dx = tf.cast(F[1][:, 0:self.domdim], dtype=np.float32)
        df = tf.cast(F[1][:, self.domdim:], dtype=np.float32)
        N = df.shape[0]
        return tf.subtract(df, self.tangentl(dfhat, dx, N))

#====================================================================
class Normalizer:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.fx_mean = None
        self.fx_std = None

    def fit(self, x, fx):
        """Compute and store normalization parameters."""
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        self.fx_mean = np.mean(fx, axis=0)
        self.fx_std = np.std(fx, axis=0)
        return self

    def transform(self, x, fx):
        """Apply normalization to x and fx."""
        x_norm = (x - self.x_mean) / self.x_std
        fx_norm = (fx - self.fx_mean) / self.fx_std
        return x_norm, fx_norm

    def inverse_transform_jacobian(self, jacobian_norm):
        """Undo normalization of Jacobian matrices."""
        nsamples = jacobian_norm.shape[0]
        J_real = np.empty_like(jacobian_norm)
        for i in range(nsamples):
            J_real[i] = np.diag(self.fx_std) @ jacobian_norm[i] @ np.diag(1.0 / self.x_std)
        return J_real

    def transform_x(self, x):
        return (x - self.x_mean) / self.x_std

    def transform_fx(self, fx):
        return (fx - self.fx_mean) / self.fx_std

    def inverse_transform_fx(self, fx_norm):
        return fx_norm * self.fx_std + self.fx_mean

    def inverse_transform_x(self, x_norm):
        return x_norm * self.x_std + self.x_mean


from tensorflow.keras.callbacks import Callback

class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        # total number of epochs is in params
        self.epochs = self.params.get('epochs', 0)
        self.pbar = tqdm(total=self.epochs, desc='Jhat Training', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        # advance one step and show the current loss
        self.pbar.update(1)
        self.pbar.set_postfix({ 'loss': f"{logs.get('loss'):.2e}" })

    def on_train_end(self, logs=None):
        self.pbar.close()