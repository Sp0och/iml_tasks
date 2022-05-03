import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split





IMG_WIDTH = 96
IMG_HEIGHT = 96
def load_image(img, training):
    '''Help Function to load the image'''
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img

def load_triplets(triplet, training):
    '''load an image triplet and stack them into a single tensor'''
    ids = tf.strings.split(triplet)
    anchor = load_image(tf.io.read_file('task4_be9ai3nsdj/food/' + ids[0] + '.jpg'), training)
    truthy = load_image(tf.io.read_file('task4_be9ai3nsdj/food/' + ids[1] + '.jpg'), training)
    falsy = load_image(tf.io.read_file('task4_be9ai3nsdj/food/' + ids[2] + '.jpg'), training)
    if training:
        return tf.stack([anchor, truthy, falsy], axis=0), 1
    else:
        return tf.stack([anchor, truthy, falsy], axis=0)

def create_model(freeze=True):
    '''Create model from previously trained preprocess NN and added layers but dont fit or predict yet'''
    # mobilenet_weights_path = 'resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #determine input shape
    inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
    #transfer model of efficient net without dense layer at the end
    encoder = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    #freeze it so that we only train our following layers
    encoder.trainable = not freeze
    decoder = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1))
    ])
    #Define nature of our output
    anchor, truthy, falsy = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
    anchor_features = decoder(encoder(anchor))
    truthy_features = decoder(encoder(truthy))
    falsy_features = decoder(encoder(falsy))
    embeddings = tf.stack([anchor_features, truthy_features, falsy_features], axis=-1)
    triple_siamese = tf.keras.Model(inputs=inputs, outputs=embeddings)
    triple_siamese.summary()
    return triple_siamese

def create_inference_model(model):
    '''Enhance model to be able to perform correct predictions'''
    distance_truthy, distance_falsy = compute_distances_from_embeddings(model.output)
    predictions = tf.cast(tf.greater_equal(distance_falsy, distance_truthy), tf.int8)
    return tf.keras.Model(inputs=model.inputs, outputs=predictions)

def compute_distances_from_embeddings(embeddings):
    '''Helper function to determine similarity of the NN output of the processed food images'''
    anchor, truthy, falsy = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_truthy = tf.reduce_sum(tf.square(anchor - truthy), 1)
    distance_falsy = tf.reduce_sum(tf.square(anchor - falsy), 1)
    return distance_truthy, distance_falsy
    
def triplet_loss(_, embeddings):
    '''Define the loss - difference between similarity of similar ones and different ones'''
    distance_truthy, distance_falsy = compute_distances_from_embeddings(embeddings)
    return tf.reduce_mean(tf.math.softplus(distance_truthy - distance_falsy))

def accuracy(_, embeddings):
    '''Calculate accuracy'''
    distance_truthy, distance_falsy = compute_distances_from_embeddings(embeddings)
    return tf.reduce_mean(
        tf.cast(tf.greater_equal(distance_falsy, distance_truthy), tf.float32))

def make_training_labels():
    '''Split the labels into training and validation dataset and return its length'''
    samples = 'task4_be9ai3nsdj/train_triplets.txt'
    with open(samples, 'r') as file:
        triplets = [line for line in file.readlines()]
    train_samples, val_samples = train_test_split(triplets, test_size=0.2, random_state=1234)
    with open('val_samples.txt', 'w') as file:
        for item in val_samples:
            file.write(item)
    with open('train_samples.txt', 'w') as file:
        for item in train_samples:
            file.write(item)
    return len(train_samples)

def make_dataset(dataset_filename, training=True):
    '''Create the dataset by replacing the image ids with the actualy loaded images'''
    dataset = tf.data.TextLineDataset(dataset_filename)
    dataset = dataset.map(
        lambda triplet: load_triplets(triplet, training),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def plot_history(history):
    '''plot the process of the classification'''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def show_batch(image_batch, label_batch):
    ''' Show Batch'''
    plt.figure(figsize=(3, 6))
    samples_idx = np.random.randint(image_batch.shape[0], size=4)
    for i in range(4):
        anchor, truthy, falsy = image_batch[samples_idx[i], ...]
        ax = plt.subplot(4, 3, 3 * i + 1)
        plt.imshow((anchor + 1.0) / 2.0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_ylabel(int(label_batch[samples_idx[i], ...]))
        ax = plt.subplot(4, 3, 3 * i + 2)
        plt.imshow((truthy + 1.0) / 2.0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax = plt.subplot(4, 3, 3 * i + 3)
        plt.imshow((falsy + 1.0) / 2.0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.show()




def main():
    #optional use argparse for cleaner arguments passed when running
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--inference_batch_size', type=int, default=256)
    parser.add_argument('--draw_results', type=bool, default=False)
    args = parser.parse_args()
    num_train_samples = make_training_labels()
    train_dataset = make_dataset('train_samples.txt')
    #create model and fit it on the train dataset
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss=triplet_loss,
                  metrics=[accuracy])
    train_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration=True) \
        .repeat().batch(args.train_batch_size)
    history = model.fit(
        train_dataset,
        steps_per_epoch=int(np.ceil(num_train_samples / args.train_batch_size)),
        epochs=args.epochs,
        validation_data=val_dataset,
        validation_steps=10
    )
    val_dataset = make_dataset('val_samples.txt')
    val_dataset = val_dataset.batch(args.train_batch_size)
    test_dataset = make_dataset('test_triplets.txt', training=False) \
        .batch(args.inference_batch_size).prefetch(2)
    #define how to predict
    inference_model = create_inference_model(model)
    num_test_samples = 59544
    predictions = inference_model.predict(
        test_dataset,
        steps=int(np.ceil(num_test_samples / args.inference_batch_size)),
        verbose=1)
    np.savetxt('predictions.txt', predictions, fmt='%i')
    if args.draw_results:
        plot_history(history)
        predictions_dataset = tf.data.TextLineDataset('predictions.txt').map(lambda t: int(t)).batch(
            args.inference_batch_size)
        test_with_predictions = tf.data.Dataset.zip((test_dataset, predictions_dataset))
        images_ids, labels = next(iter(test_with_predictions))
        show_batch(images_ids.numpy(), labels.numpy())


if __name__ == '__main__':
    main()
