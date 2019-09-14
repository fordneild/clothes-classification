# TensorFlow and tf.karas
import tensorflow as tf
from tensorflow import keras

#helpers
import numpy as np
import matplotlib.pyplot as plt

#check version
#print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

#load data sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Image and label testing
#print(train_images[0][4][12])


#Clothes types
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

firstImage = plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

train_images = train_images / 255.0
test_images = test_images / 255.0

labeledImages = plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([
    #Flatten 28 x 28 image into 784 imput neurons
    keras.layers.Flatten(input_shape=(28, 28)),
    #One layer with 128 neurons
    keras.layers.Dense(128, activation=tf.nn.relu),
    #Output layer
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#setting for back propagation and fitness test
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the data
model.fit(train_images, train_labels, epochs=5)

#tests the data with out test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#lets get label predictions for all of our test data
#is an array of 10 numbers corresponding to our labels
predictions = model.predict(test_images)

#which one of the 10 is heighest (whats its best guess for the first image)
print(np.argmax(predictions[0]))

def plot_image(i, predictions_array, true_label, img):
  #set variables for our chosen index
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Get label and predictions for i index in test data
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# Grab an image from the test dataset
img = test_images[0]

#expand it into a collection (cause that is what keras likes)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

#an array of item predictions (only one)
predictions_single = model.predict(img)

#for the first image plot its prediction
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


plt.show()
