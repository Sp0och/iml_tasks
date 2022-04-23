import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

features = pd.read_csv("input_data/train_features.csv")
labels = pd.read_csv("input_data/train_labels.csv")

vital_signs = ['RRate', 'ABPm' , 'SpO2', 'Heartrate']
vital_signs_LABELS = ['LABEL_RRate', 'LABEL_ABPm' , 'LABEL_SpO2', 'LABEL_Heartrate']

# sortiere nach pid, einfach um sicher zu gehen, inplace: verändert features direkt, ansonsten würde
# er es returnen und du müsstes eine neue variable definieren
#features.sort_values(by='pid', inplace=True)
#labels.sort_values(by='pid',inplace=True)

# unique nimmt alle pid nur einmal, damit du alle einmal hast
pids = features["pid"].unique()

# test Laufvariable, von LABEL_RRate bis LABEL_Heartrate
# gehe durch beide laufvariablen parallel durch
for test, label in zip(vital_signs, vital_signs_LABELS):
  #initailisiere für jeden test ein leeres array, runde klammern da funktion
  x_train = np.empty([len(pids),12])
  y_train = labels[label].to_numpy()
  #print(x_train.shape)
  # gehe nun durch alle pids druch, in: du gehst durch dein array mit allen pids durch!
  # enumerate gibt pid und den ort wo es steht
  for idx, pid in enumerate(pids):
    #mache das array ready und schreibe es danach in den x_train rein
    # iteriere durch alle pid durch. test wird nur in der äusseren spalte geändert
    # ohne to_numpy ist es ein stehender vektor mit 1 2 3 eintrag pro pid
    #print(features[features["pid"] == pid])
    #print(label)
    # innere klammer wird 12 mal true sein, sonst false. features[true] liest den wert raus und schriebt 
    # in patient data, das ist ein zwischenkonstrukt
    patient_data = features[features["pid"] == pid][test].to_numpy()
    #if pid == 1:
    #  print(patient_data)
    # Hilfskonstrukt: Gib dir pid raus, wo ausschlisslich nan werte vorkommen
    if len(patient_data[~np.isnan(patient_data)]) == 0:
      # für Patienten mit ausschliesslich means, nehmen wir den mean der gesamtbevölkerung
      # TODO replace with Mean instead of zeros
      patient_data = np.zeros(12)
    else: 
      # tilde ist negierung, dort wo nicht nan ist->bilde den mean
      mean = np.mean(patient_data[~np.isnan(patient_data)])
      #np.isnan gibt ein array raus was genau dort true ist, wo patient_data einen nan hat. das wollen wir accessen
      patient_data[np.isnan(patient_data)] = mean
    # accesse xtrain genau in der zeile dank enumerate mit idx
    x_train[idx,:] = patient_data
  # bevor wir den fit machen, splitte data set zum validieren
  split_idx = int(len(x_train)*0.9)
  # funktion gibt zwei arrays raus, 
  #split = np.array_split(x_train,split_idx)
  x_train_split,x_valid,y_train_split,y_valid = train_test_split(x_train,y_train, train_size=0.9)
  # in der selben schleife die durch tests geht->fitte die daten
  regressor = svm.LinearSVR()
  regressor.fit(x_train_split,y_train_split)
  # y wollen wir damit nun predicten, mit y_valid checken wir das nachher
  y_predict = regressor.predict(x_valid)
  error = np.sqrt(np.mean((y_predict-y_valid)**2))
  print(error)