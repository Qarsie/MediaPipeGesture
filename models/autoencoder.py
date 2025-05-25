import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

def build_autoencoder(input_dim=63, latent_dim=32):
    # Encoder
    inputs = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(inputs)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(latent_dim, activation='linear', name='bottleneck')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Models
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    
    return autoencoder, encoder

def compile_autoencoder(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model