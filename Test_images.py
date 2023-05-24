import os
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import keras.utils as image
from keras.utils import img_to_array



TESTING_DIR = "TestSamples"
test_images = []
for i in os.listdir(TESTING_DIR):
    img_path = TESTING_DIR + '/' + i
    #print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    test_images.append(img_array)
test_images = np.array(test_images)


model = load_model("Resnet_model.h5")

prediction = model.predict(test_images)


label = ['Basketball','Football','Rowing', 'Swimming','Tennis', 'Yoga']
predicted_labels = []


for i in range(len(test_images)):
  pos = np.argmax(prediction[i], axis=0)
  #print(prediction[i])
  #print(pos)
  image_label = label[pos]

  if image_label == "Basketball":
    predicted_labels.append(0)
  elif image_label == "Football":
    predicted_labels.append(1)
  elif image_label == "Rowing":
    predicted_labels.append(2)
  elif image_label == "Swimming":
    predicted_labels.append(3)
  elif image_label == "Tennis":
    predicted_labels.append(4)
  elif image_label == "Yoga":
    predicted_labels.append(5)

print(predicted_labels)



print(os.listdir(TESTING_DIR))

dataframe_answer = pd.DataFrame({'image_name':  os.listdir(TESTING_DIR) , 'label': predicted_labels})
dataframe_answer.to_csv('BIO55.csv', index=False)
