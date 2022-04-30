from xml.parsers.expat import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

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


class model_manager:
    def __init__(self,img_height,img_width):
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        #create data augmentation by randomly switching horizontally and randomly rotate by 0.1
        self.inputs = tf.keras.Input(shape=(3, self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )
        norm_layer = tf.keras.layer.experimental.preprocessing.Normalization()
        mean = np.array([127.5]*3)
        var = mean**2
        encoder =  tf.keras.applications.EfficientNetB4(
        include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), weights="imagenet")
        encoder.trainable = False
        decoder = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Lambda(
                lambda t: tf.math.l2_normalize(t, axis=1))
        ])
        image, similar, different = self.inputs[:, 0, ...], self.inputs[:, 1, ...], self.inputs[:, 2, ...]
        image_normed = norm_layer(data_augmentation(image))
        similar_normed = norm_layer(data_augmentation(similar))
        different_normed = norm_layer(data_augmentation(different))
        norm_layer.set_weights([mean,var])
        image_features = decoder(encoder(image_normed))
        similar_features = decoder(encoder(similar_normed))
        different_features = decoder(encoder(different_normed))
        self.outputs = tf.stack([image_features,similar_features,different_features],axis=-1)
        self.model = tf.keras.Model(inputs=self.inputs,outputs=self.outputs)
        #create the basic model with the augmentation, encoder and decoder layers
    def loss_function(self):
        print("hello world")
    def metric(self):
        print("hello world")
    def set_optimizer(self):
        print("hello world")
    def fit(self):
        print("hello world")
    def add_predictor(self):
        print("hello world")

#preview of data augmentation
# image = plt.imread("00013.jpg")
# plt.figure(figsize=(10,10))
# data_augmentation = tf.keras.Sequential(
#     [
#         tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
#         tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
#     ]
# )
# for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = data_augmentation(
#             tf.expand_dims(image, 0), training=True
#         )
#         plt.imshow(augmented_image[0].numpy().astype("int32"))
#         plt.axis("off")
# plt.show()
IMG_HEIGHT = 256
IMG_WIDTH = 256
#create data augmentation by randomly switching horizontally and randomly rotate by 0.1
inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)
norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5]*3)
var = mean**2
encoder =  tf.keras.applications.EfficientNetB4(
include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), weights="imagenet")
encoder.trainable = False
decoder = tf.keras.Sequential([
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1))
])
image, similar, different = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
image_normed = norm_layer(data_augmentation(image))
similar_normed = norm_layer(data_augmentation(similar))
different_normed = norm_layer(data_augmentation(different))
# norm_layer.set_weights([mean,var])
image_features = decoder(encoder(image_normed))
similar_features = decoder(encoder(similar_normed))
different_features = decoder(encoder(different_normed))
outputs = tf.stack([image_features,similar_features,different_features],axis=-1)
model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()