import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

################# Parameters #####################
path = "myData"  # folder with all the class folders
labelFile = 'labels.csv'  # file with all names of classes
batch_size_val = 50  # how many to process together
epochs_val = 20
imageDimensions = (32, 32, 3)  # Adjust the dimensions as needed
testRatio = 0.2  # if 1000 images split will 200 for testing
validationRatio = 0.2  # if 1000 images 20% of remaining 800 will be 160 for validation
###################################################

# Importing images and labels
images, classNo = [], []
myList = os.listdir(path)
noOfClasses = len(myList)

print("Total Classes Detected:", noOfClasses)
print("Importing Classes.....")
for count in range(noOfClasses):
    myPicList = os.listdir(os.path.join(path, str(count)))
    for y in myPicList:
        curImg = cv2.imread(os.path.join(path, str(count), y))
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))  # Resize images
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
print(" ")

images = np.array(images)
classNo = np.array(classNo)

############################### Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

############################### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("Data Shapes")
print(f"Train: {X_train.shape, y_train.shape}")
print(f"Validation: {X_validation.shape, y_validation.shape}")
print(f"Test: {X_test.shape, y_test.shape}")

assert X_train.shape[0] == y_train.shape[0], "The number of images is not equal to the number of labels in the training set"
assert X_validation.shape[0] == y_validation.shape[0], "The number of images is not equal to the number of labels in the validation set"
assert X_test.shape[0] == y_test.shape[0], "The number of images is not equal to the number of labels in the test set"
assert X_train.shape[1:] == imageDimensions, "The dimensions of the training images are wrong"
assert X_validation.shape[1:] == imageDimensions, "The dimensions of the validation images are wrong"
assert X_test.shape[1:] == imageDimensions, "The dimensions of the test images are wrong"

############################### READ CSV FILE
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

############################## DISPLAY SOME SAMPLES IMAGES  OF ALL THE CLASSES
num_classes = noOfClasses
cols = 2
num_of_samples = []

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(2, num_classes * 5))
fig.tight_layout()

for j in range(num_classes):
    x_selected = X_train[y_train == j]
    num_of_samples.append(len(x_selected))
    random_indices = np.random.choice(len(x_selected), min(cols, len(x_selected)), replace=False)

    axs[j][0].imshow(x_selected[random_indices[0]], cmap='gray')
    axs[j][0].axis('off')
    axs[j][1].text(0.5, 0.5, f"{j}-{data.iloc[j]['Name']}", fontsize=10, ha='center', va='center')
    axs[j][1].axis('off')

############################### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

############################### PREPROCESSING THE IMAGES
def preprocess_images(images):
    processed_images = np.array([cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255 for img in images])
    return processed_images.reshape(processed_images.shape[0], processed_images.shape[1], processed_images.shape[2], 1)

X_train = preprocess_images(X_train)
X_validation = preprocess_images(X_validation)
X_test = preprocess_images(X_test)

############################### AUGMENTATION OF IMAGES
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# Show augmented images
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]), cmap='gray')
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

############################### CONVOLUTION NEURAL NETWORK MODEL
def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

############################### TRAIN
model = myModel()
print(model.summary())

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    epochs=epochs_val,
                    steps_per_epoch=len(X_train) // batch_size_val,
                    validation_data=(X_validation, y_validation))

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# STORE THE MODEL AS A PICKLE OBJECT
with open("model_trained.p", "wb") as pickle_out:
    pickle.dump(model, pickle_out)

cv2.waitKey(0)
