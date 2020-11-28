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
from skimage.transform import resize


myParser = ImageParser()
model = tf.keras.models.load_model('model1')

testRes = np.array([-8.32, 4.1])
ABStestRes = abs(testRes)
print(ABStestRes)
print(np.amax(ABStestRes))
#classPred = class_names[np.where(ABStestRes == np.amax(ABStestRes))]

#for i in range(2962,3000):
#for i in range(500):
#    img=cv2.imread(list(myParser.imageDict.values())[i].fileName)
#    print("The label for this image is", list(myParser.imageDict.values())[i].label)
#    resize_img = resize(img, (256,256,3))
#    plt.imshow(resize_img)
#    plt.show()
#    predictions = np.argmax(model.predict(np.array([resize_img])), axis=-1)
#    print(predictions)
    #smax = tf.nn.softmax(predictions)
    #print(smax)
    #argmaxPred = np.argmax(smax)
    #print(argmaxPred)


#The first chunk here that reads in the files initially is just so we have a validation training set to check our results against to find the images that were misclassified
data_dir = "C:\\Users\\crsny\\Documents\\GradSchool\\2020-2021\\Project\\DurOrNoDur\\ImageFiles2"

batch_size = 32
img_height = 256
img_width = 256

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

print("size of val_ds when imported: ", len(val_ds))
val_ds_images_labels = val_ds.take(1)
print("size of val_ds take(1): ", len((val_ds_images_labels)))

class_names = train_ds.class_names
print("The names of the classes are: ", class_names)
print(class_names[0])
print(np.where(ABStestRes == np.amax(ABStestRes)))
print(ABStestRes.tolist().index(np.amax(ABStestRes)))
#result = np.where(ABStestRes == np.amax(ABStestRes))
resultIDX = ABStestRes.tolist().index(np.amax(ABStestRes))

classPred = class_names[resultIDX]
print(classPred)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")



AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

results = model.evaluate(val_ds)
print("test loss, test accuracy: ", results)

valResults = np.genfromtxt("C:\\Users\\crsny\\Documents\\GradSchool\\2020-2021\\Project\\DurOrNoDur\\predictionResults_v0.csv", delimiter=",")
print(valResults[1])
print(valResults[2])

#outputResults=[]
print("length of valResults (classified results) is: ", len(valResults))
print("length of val_ds (initial dataset and actual class) is: ", len(val_ds.take(1)))
print("val_ds __len__(): ", val_ds.__len__())


#outputResults = np.array([[valResults],[val_ds.labels]])
#outputResults[1,1]

#outputResults[:,1] = valResults
#outputResults[:,2] = val_ds.labels

# see if the lengths of valResults and val_ds is the same

counter = 0
incorrectClass = 0
correctClass = 0
#outputResults[0,counter]
#dataset.range(881)

iterator = iter(val_ds)
print(iterator.get_next())
print(iterator.get_next())

#for the number of elements in the results iterate over them and if they do not agree, print out the corresponding image
for images, labels in val_ds:
    print("The correct label for this image is: ", class_names[labels[counter]])
    if class_names[labels[counter]] == "Dur":
        trueClass = 0
    if class_names[labels[counter]] == "NoDur":
        trueClass = 1
    
    print("The classified label for this image is: ", valResults[counter])
    #currentPredictedClass = int(valResults[counter], 10)

    if trueClass != valResults[counter]:
        #plt.figure(2)
        #plt.imshow(images[counter].numpy().astype("uint8"))
        #plt.show()
        incorrectClass+=1
    else:
        correctClass+=1


    counter+=1



np.savetxt("C:\\Users\\crsny\\Documents\\GradSchool\\2020-2021\\Project\\DurOrNoDur\\testing\\predictionResultsAppend_v0.csv", valResults, delimiter=",")