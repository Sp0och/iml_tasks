from predictor import predict
from trainer import train
import torch

# split train dataset into 10 equal parts

# repeat leaving out 1 different fold each time:
    # find weights with 9 parts of train data

    # find RMSE of remaining part

# average over all the RMSE's

# repeat for different lambda

train_file = "train.csv"
validation_file = "test.csv"
model_name = "linear2" #linear2 was used for submission and is the one with the seed currently hardcoded

if __name__ == "__main__":
    # if the model does not exist yet, train it
    print(f"Looking for model: {model_name}")
    try:
        model = torch.load(model_name)
        print("Found model")
    except:
        print("Creating new model...")
        model = train(annotations_file=train_file, model_name=model_name)

    print("Start prediction...")
    predict(validation_file, model)