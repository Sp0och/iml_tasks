from xml.parsers.expat import model
import tensorflow as tf
import numpy as np

#choose minimum and resize accordingly before inputting
IMG_HEIGHT = 256
IMG_WIDTH = 256

def create_model(freeze=True):
    '''Create model from previously trained preprocess net and addded layers\n
    No fitting, compiling or predicting in this function merely the creation of the structure'''
    #determine input shape
    inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
    #transfer model of efficient net without dense layer at the end
    encoder = tf.keras.applications.EfficientNetB4(
        include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), weights="imagenet")
    #freeze it so that we only train our following layers
    encoder.trainable = not freeze
    #add our following layers TODO: change these to match output of Efficient net and play around
    decoder = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1))
    ])
    #create the definition of our output
    anchor, similar, different = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
    anchor_features = decoder(encoder(anchor))
    similar_features = decoder(encoder(similar))
    different_features = decoder(encoder(different))
    embeddings = tf.stack([anchor_features, similar_features, different_features], axis=-1)
    #create the model from the knowledge of our desired inputs and output
    model = tf.keras.Model(inputs=inputs, outputs=embeddings)
    model.summary()
    return model
