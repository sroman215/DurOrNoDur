import numpy as np
import matplotlib
from imageParser import ImageParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import cv2
from scipy import misc
#import Pillow as PIL
from PIL import Image
#import skimage
from skimage.transform import resize

#scipy.misc.imresize

# SAR TODO - Move this code into the Classifier class

myParser = ImageParser()


#for i in range(447):
#    img=cv2.imread(list(myParser.imageDict.values())[i].fileName)
#    print("The label for this image is", list(myParser.imageDict.values())[i].label)
#    resize_img = resize(img, (256,256,3))
    
    


print(list(myParser.imageDict.values())[0].fileName)

img=cv2.imread(list(myParser.imageDict.values())[3740].fileName)
imgplot = plt.imshow(img)
plt.show()
plt.imshow(img)

height, width, channels = img.shape
print("width: ", width)
print("height: ", height)
#new_shape = ()
print("The number of images are: ", len(myParser.imageDict))
print("The classification of the first image is: ", list(myParser.imageDict.values())[0].label)
#search through the images, find the smallest resolution and scale all of the images to that resolution
#newShape = ()

imgPIL = Image.open(list(myParser.imageDict.values())[0].fileName)
myParser.getImageFilesDir()

smallIMG = imgPIL.resize((250,250), Image.ANTIALIAS)
#plt.imshow(smallIMG)
#plt.show()

data_dir = "C:\\Users\\crsny\\Documents\\GradSchool\\2020-2021\\Project\\DurOrNoDur\\ImageFiles2"

batch_size = 32
img_height = 256
img_width = 256

#test_file = tf.keras.utils.get_file(list(myParser.imageDict.values())[0].fileName)
test_img = tf.keras.preprocessing.image.load_img(list(myParser.imageDict.values())[0].fileName, target_size=[img_height, img_width])
nparray_test_img = np.array(test_img)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(img_height, img_width), shuffle=True, seed=123,
    validation_split=0.2, subset="training", interpolation='bilinear', follow_links=False
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print("The names of the classes are: ", class_names)
print(class_names[0])

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#testing the unshuffled test read
data_dir_test = "C:\\Users\\crsny\\Documents\\GradSchool\\2020-2021\\Project\\DurOrNoDur\\ImageFilesTesting"
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_test, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(256, 256), shuffle=False, interpolation='bilinear', follow_links=False
)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]


num_classes = 2

#This creates a concolution kernel convolved with the layer input to produce a tensor of outputs
#The first two numbers dictate the dimentionality of the output spce (the number of output filters in the convolution)
#   and the height/width of the 2D convolution window
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=6
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#model.save('"C:\\Users\\crsny\\Documents\\GradSchool\\2020-2021\\Project\\DurOrNoDur\\model1"')
#model.save('model1')

valResults = model.predict_classes(val_ds)


np.savetxt("predictionResults_v0p1.csv", valResults, delimiter=",")

results = model.evaluate(val_ds)
print("test loss, test accuracy: ", results)
predictedResults = model.predict(val_ds)

data_dir_test = "C:\\Users\\crsny\\Documents\\GradSchool\\2020-2021\\Project\\DurOrNoDur\\ImageFilesTesting"

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_test, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=32, image_size=(img_height, img_width), shuffle=False, interpolation='bilinear', follow_links=False
)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()

test_ds_pred = model.predict(test_ds)
test_pred_classes = model.predict_classes(test_ds)
print(test_ds_pred)
np.savetxt("predictionResults_testDS_v0p0.csv", test_ds_pred, delimiter=",")
np.savetxt("predictionResults_testDS_Classes_v0p0.csv", test_pred_classes, delimiter=",")
test_ds_eval = model.evaluate(test_ds)
print(test_ds_eval)


#This doesn't work since the input is expecting a batch size of 32
test_img = tf.keras.preprocessing.image.load_img(list(myParser.imageDict.values())[0].fileName, target_size=[img_height, img_width])
test_img_np = np.array(test_img)

test_img_pred = model.predict(np.array(test_img))
print("Test img prediction results: ", test_img_pred)

test_img2 = tf.keras.preprocessing.image.load_img(list(myParser.imageDict.values())[3000].fileName, target_size=[img_height, img_width])

test_img_pred2 = model.predict(test_img2)
print("Test img2 prediction results: ", test_img_pred2)

#model.save('model1')

#trying classification to test things
#take absolute value of classification
#count the number of classifications for each
# if the classification does not match save the filename


correctCounter = 0
incorrectCounter = 0
for i in range(100):
    img=cv2.imread(list(myParser.imageDict.values())[i].fileName)
    print("The label for this image is", list(myParser.imageDict.values())[i].label)
    resize_img = resize(img, (256,256,3))
    predictions = model.predict(np.array([resize_img]))
    #testRes = np.array([-8.32, 4.1])
    absPred = abs(predictions)
    #print(absPred)
    #print(np.amax(absPred))
    print("raw prediction values: ", predictions)
    #classPred = class_names[np.where(absPred == np.amax(absPred))]
    resultIDX = absPred.tolist().index(np.amax(absPred))
    classPred = class_names[resultIDX]
    print("The classification of this image is: ", classPred)
    if resultIDX == list(myParser.imageDict.values())[i].label:
        correctCounter+=1
    elif resultIDX != list(myParser.imageDict.values())[i].label:
        incorrectCounter+=1
    
print("The number of correct classifications is: ", correctCounter)
print("The number of incorrect classifications is: ", incorrectCounter)


#for i in range(447):
correctCounter = 0
incorrectCounter = 0
for j in range(2962,3000):
    img=cv2.imread(list(myParser.imageDict.values())[j].fileName)
    print("The label for this image is", list(myParser.imageDict.values())[j].label)
    resize_img = resize(img, (256,256,3))
    predictions = model.predict(np.array([resize_img]))
    #testRes = np.array([-8.32, 4.1])
    absPred = abs(predictions)
    #print(absPred)
    #print(np.amax(absPred))
    print("raw prediction values: ", predictions)
    #classPred = class_names[np.where(absPred == np.amax(absPred))]
    resultIDX = absPred.tolist().index(np.amax(absPred))
    classPred = class_names[resultIDX]
    print("The classification of this image is: ", classPred)
    if resultIDX == list(myParser.imageDict.values())[j].label:
        correctCounter+=1
    elif resultIDX != list(myParser.imageDict.values())[j].label:
        incorrectCounter+=1
    
print("The number of correct classifications is: ", correctCounter)
print("The number of incorrect classifications is: ", incorrectCounter)

#search through the predicted results and find some of the values that are incorrectly identified and print them....
#plt.figure()
#for i in len(valResults):
    #if
