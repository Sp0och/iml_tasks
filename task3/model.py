from xml.parsers.expat import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

#choose minimum image size and resize accordingly before inputting

class model_manager:
    def __init__(self,img_height,img_width):
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.inputs = tf.keras.Input(shape=(3, self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        #create data augmentation by randomly switching horizontally and randomly rotate by 0.1
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )
        #EfficientNet does not need preprocessing and expects inputs in tensor form in the range 0-255
        encoder =  tf.keras.applications.EfficientNetB4(
            include_top=False, input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3), 
            weights="imagenet")
        encoder.trainable = False
        decoder = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Lambda(
                lambda t: tf.math.l2_normalize(t, axis=1))
        ])
        image, similar, different = self.inputs[:, 0, ...], self.inputs[:, 1, ...], self.inputs[:, 2, ...]
        image_features = decoder(encoder(data_augmentation(image)))
        similar_features = decoder(encoder(data_augmentation(similar)))
        different_features = decoder(encoder(data_augmentation(different)))
        self.out_features = tf.stack([image_features,similar_features,different_features],axis=-1)
        self.model = tf.keras.Model(inputs=self.inputs,outputs=self.out_features)
        print("Successfully built basic model!")

    def calc_difference(self, y_pred):
        '''Compute the difference between the considered image and the alledged 
        similar and different one respectively. Returns two positive numbers'''
        image = y_pred[...,0]
        similar = y_pred[...,1] 
        different = y_pred[...,2]
        # sum of squared differences
        sim_diff = tf.reduce_sum(tf.square(image - similar),1)
        dif_diff = tf.reduce_sum(tf.square(image - different),1)
        return sim_diff, dif_diff

    def loss_function(self, y_pred):
            '''loss function to minimize during back propagation'''
            sim_diff, dif_diff = self.calc_difference(y_pred)
            #consider the cases where sim_diff is significantly higher than dif_diff
            loss = tf.reduce_mean(tf.math.softmax(sim_diff-dif_diff))
            return loss

    def compile(self):
        print("Compilation Initiated...")
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss=self.loss_function,
                            metrics=self.loss_function)
        print("NN compiled successfully!")

    def fit(self,_train_ds, _valid_ds,  _epochs=1, _batch_size=32,_verbose=1):
        #59515 being the length of the training samples
        print(f"Batch size: {_batch_size}")
        print(f"Epochs: {_epochs}")
        print(f"Steps: {np.ceil(47612/_batch_size)-1}")
        print(f"Starting fitting procedure...")
        self.model.fit(_train_ds,batch_size=_batch_size, epochs=_epochs,validation_data=_valid_ds,verbose=_verbose, steps_per_epoch=int(np.ceil(59515*0.8/_batch_size)-1))

    def add_predictor(self):
        sim_diff, dif_diff = self.calc_difference(self.out_features)
        prediction = tf.cast(tf.greater_equal(dif_diff,sim_diff),np.int8)
        self.model = tf.keras.model(inputs=self.input,outputs=prediction)

def create_base_model():
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    #create data augmentation by randomly switching horizontally and randomly rotate by 0.1
    inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )
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
    image_features = decoder(encoder(data_augmentation(image)))
    similar_features = decoder(encoder(data_augmentation(similar)))
    different_features = decoder(encoder(data_augmentation(different)))
    outputs = tf.stack([image_features,similar_features,different_features],axis=-1)
    model = tf.keras.Model(inputs=inputs,outputs=outputs)
    print("Successfully built basic model!")

def loss_function():
    '''loss function to minimize during back propagation'''
    sim_diff, dif_diff = calc_difference()
    #consider the cases where sim_diff is significantly higher than dif_diff
    loss = tf.reduce_mean(tf.math.softmax(sim_diff-dif_diff))
    return loss

def calc_difference(outputs):
    '''Compute the difference between the considered image and the similar and different one respectively\n
    Returns two positive numbers'''
    image, similar, different = outputs[...,0], outputs[...,1], outputs[...,2]
    sim_diff = tf.reduce_sum(tf.square(image - similar),1)
    dif_diff = tf.reduce_sum(tf.square(image - different),1)
    return sim_diff, dif_diff

def compile(self):
    print("Compilation Initiated...")
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=self.loss_function,
                        metrics=self.loss_function)
    print("This NN compiled!!!")
def fit(self,_train_ds,_epochs,_validation_split=0.15,_verbose=2):
    self.model.fit(_train_ds,epochs=_epochs,validation_split=_validation_split,verbose=_verbose)

def add_predictor(self):
    sim_diff, dif_diff = self.calc_difference()
    prediction = tf.cast(tf.greater_equal(dif_diff,sim_diff),np.int8)
    self.model = tf.keras.model(inputs=self.input,outputs=prediction)



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
if __name__ == "__main__":
  with tf.device('/device:GPU:0'):
    manager = model_manager(256,256)
    # manager.model.summary()
    # manager.compile()
    # manager.fit(train_dataset, val_dataset)
    # manager.add_predictor()
    # sim_diff, dif_diff = manager.calc_difference(manager.out_features)
    # prediction = tf.cast(tf.greater_equal(dif_diff,sim_diff),np.int8)
    # manager.model = tf.keras.model(inputs=manager.input,outputs=prediction)
    # predictions = manager.model.predict(test_dataset,verbose=1)
    # np.savetxt(path_to_folder + 'predictions.txt', predictions,fmt='%i')