import preprocess_data
import os.path
import task_1_2
import pandas as pd

if __name__ == "__main__":
  # preprocess input data
  print("[Starting] Preprocessing input data")
  if not os.path.isfile("output_data/test_features_processed.csv") \
      or not os.path.isfile("output_data/train_features_processed.csv"):
    preprocess_data.DataProcessor()
  print("[Finished] Preprocessing input data")

  # train model
  print("[Starting] Training model")
  feature_df = pd.read_csv("output_data/train_features_processed.csv")
  label_df = pd.read_csv("input_data/train_labels.csv")
  clf = task_1_2.train_model(feature_df, label_df)
  print("[Finished] Training model")

  # predict on test data
  print("[Starting] Predicting on test data")
  feature_df = pd.read_csv("output_data/test_features_processed.csv")
  predicted_data = task_1_2.predict_on_test_data(clf, feature_df)
  print("[Finished] Predicting on test data")
  # write to sample csv
  print("[Starting] Writing to sample csv")
  predicted_data.to_csv("output_data/submission.csv", index=False)
  print("[Finished] Writing to sample csv")


