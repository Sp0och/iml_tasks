import preprocess_data
import os.path
import task_1_2
import task_3
import task_3_trend
import pandas as pd
import pickle

if __name__ == "__main__":
  # TODO create folders if they don't exist yet

  # preprocess input data
  print("[Starting] Preprocessing input data")
  if not os.path.isfile("output_data/test_features_processed.csv") \
      or not os.path.isfile("output_data/train_features_processed.csv"):
    preprocess_data.DataProcessor()
  print("[Finished] Preprocessing input data")

  # train model
  print("[Starting] Training model")
  label_df = pd.read_csv("input_data/train_labels.csv")
  if not os.path.isfile("models/model"):
    feature_df = pd.read_csv("output_data/train_features_processed.csv")
    clf_dict = task_1_2.train_model(feature_df, label_df)
    pickle.dump(clf_dict, open("models/model", 'wb'))
  else:
   clf_dict = pickle.load(open("models/model", 'rb'))
  # feature_df = pd.read_csv("output_data/train_reg_features.csv")
  # reg_dict = task_3.train_model(feature_df, label_df)
  feature_df = pd.read_csv("output_data/train_features_processed.csv")
  reg_dict = task_3_trend.train_model(feature_df, label_df)
  print("[Finished] Training model")

  # predict on test data
  print("[Starting] Predicting on test data")
  feature_df = pd.read_csv("output_data/test_features_processed.csv")
  predicted_data_1_2 = task_1_2.predict_on_test_data(clf_dict, feature_df)
  # feature_df = pd.read_csv("output_data/test_reg_features.csv")
  # predicted_data_3 = task_3.predict_on_test_data(reg_dict, feature_df)
  feature_df = pd.read_csv("output_data/test_features_processed.csv")
  predicted_data_3 = task_3_trend.predict_on_test_data(reg_dict, feature_df)
  print("[Finished] Predicting on test data")
  # write to sample csv
  print("[Starting] Writing to sample csv")
  predicted_data = predicted_data_1_2.merge(predicted_data_3, on='pid')
  predicted_data.to_csv("output_data/submission.csv", float_format='%.6f', index=False)
  print("[Finished] Writing to submission csv")


