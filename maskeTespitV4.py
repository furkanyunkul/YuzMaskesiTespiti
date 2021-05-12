#Sonuc Cikti

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import time

#Veri uzantisi
dataset=r'D:\yapayZeka\covid\experiements\data'
imagePaths=list(paths.list_images(dataset))
#print(imagePaths)

#eğitim,validation ve test yapma
trainList,testList=train_test_split(imagePaths, test_size=0.4, random_state=42)
valList, testList = train_test_split(testList, test_size=0.5, random_state=42)

print('total =',len(imagePaths))
print('train :',len(trainList))
print('val   :',len(valList))
print('test  :',len(testList))

#data vektoru


#data kısmında vektorleri bulma
#labels kısmında maske olup olmadıgını bulma
#-*****************************************************************************
#Train icin 

dataTrain=[]
labelsTrain=[]

for i in trainList:
    label=i.split(os.path.sep)[-2]
    labelsTrain.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    dataTrain.append(image)

"""    
print(dataTrain)
print(labelsTrain)
"""

#data ve labelsı arraye donusturme
dataTrain=np.array(dataTrain,dtype='float32')
labelsTrain=np.array(labelsTrain)

"""    
print(dataTrain)
print(labelsTrain)
"""
#Labelsları 1-0 encode donusturme
lb = LabelBinarizer()
labelsTrain = lb.fit_transform(labelsTrain)
print("Text categories in number form: \n",labelsTrain)
labelsTrain = to_categorical(labelsTrain)
print("One Hot Encoding: \n",labelsTrain)

#toplam lable boyutu
print(labelsTrain.shape)

#Resim gosterme
plt.imshow(dataTrain[400])

#**********************************************************************************
#Validation icin

dataVal=[]
labelsVal=[]

for i in valList:
    label=i.split(os.path.sep)[-2]
    labelsVal.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    dataVal.append(image)

"""    
print(dataVal)
print(labelsVal)
"""

#data ve labelsı arraye donusturme
dataVal=np.array(dataVal,dtype='float32')
labelsVal=np.array(labelsVal)

"""    
print(dataVal)
print(labelsVal)
"""
#Labelsları 1-0 encode donusturme
lb = LabelBinarizer()
labelsVal = lb.fit_transform(labelsVal)
print("Text categories in number form: \n",labelsVal)
labelsVal = to_categorical(labelsVal)
print("One Hot Encoding: \n",labelsVal)

#toplam lable boyutu
print(labelsVal.shape)

#Resim gosterme
plt.imshow(dataVal[100])

#**********************************************************************************
#Test icin

dataTest=[]
labelsTest=[]

for i in testList:
    label=i.split(os.path.sep)[-2]
    labelsTest.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    dataTest.append(image)

"""    
print(dataTest)
print(labelsTest)
"""

#data ve labelsı arraye donusturme
dataTest=np.array(dataTest,dtype='float32')
labelsTest=np.array(labelsTest)

"""    
print(dataVal)
print(labelsVal)
"""
#Labelsları 1-0 encode donusturme
lb = LabelBinarizer()
labelsTest = lb.fit_transform(labelsTest)
print("Text categories in number form: \n",labelsTest)
labelsTest = to_categorical(labelsTest)
print("One Hot Encoding: \n",labelsTest)

#toplam lable boyutu
print(labelsTest.shape)

#Resim gosterme
plt.imshow(dataTest[20])

#******************************************************************************

#Train versis degistirme

train_generator = ImageDataGenerator(
                    rotation_range = 20,
                    zoom_range = 0.15,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.15,
                    horizontal_flip = True,
                    fill_mode="nearest"
                  )

#MobileNetV2 modeli eklendi

#names = [ 'InceptionV3', 'MobileNet', 'DenseNet', 'VGG19',"MobileNetV2"]

baseModel = MobileNetV2(input_tensor = tf.keras.layers.Input(shape = (224,224,3)),include_top = False, weights = 'imagenet')
#baseModel = InceptionV3(input_tensor = tf.keras.layers.Input(shape = (224,224,3)),include_top = False)
#baseModel = DenseNet201(input_tensor = tf.keras.layers.Input(shape = (224,224,3)),include_top = False)

print(baseModel)

print(baseModel.output)
headModel = baseModel.output              # headModel input headNodel ouputu
#7,7 yerine 2,2 yapıldı
headModel = tf.keras.layers.AveragePooling2D(pool_size = (2,2))(headModel)
headModel = tf.keras.layers.Flatten()(headModel)
headModel = tf.keras.layers.Dense(128, activation = 'relu')(headModel)
headModel = tf.keras.layers.Dropout(0.5)(headModel)
headModel = tf.keras.layers.Dense(2, activation = 'softmax')(headModel)



init_lr = 1e-4  # 0.0001
epochs = 10
bs = 32



model = tf.keras.models.Model(inputs = baseModel.input, outputs = headModel)

#model özeti
model.summary()




for layers in baseModel.layers:
  baseModel.trainable = False
  
#Derleme

opt = Adam(lr = init_lr, decay = init_lr/epochs)
model.compile(optimizer=opt, loss = 'binary_crossentropy',metrics = ['accuracy']) 
  
  
"""## Fitting the model"""
history = model.fit(train_generator.flow(dataTrain,labelsTrain,batch_size=bs),
                    steps_per_epoch = len(dataTrain)//bs,
                    validation_data = (dataVal,labelsVal),
                    validation_steps = len(dataVal)//bs,
                    epochs = epochs,
                    verbose = 1
                    ) 

#Sonuc
print(model.evaluate(dataTest,labelsTest))

#Precision Recall f1Score 
print("Precision Recall f1Score ")
predIdxs = model.predict(dataTest, batch_size=32)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(labelsTest.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


#Dogruluk ve Kayıp Degerleri
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
  
#Dogruluk ve Kayıp Grafikleri
plt.plot(epochs, acc, 'b', label='Eğitim Doğruluğu')
plt.plot(epochs, val_acc, 'r', label='Validation Doğruluğu')
plt.title('Model Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk Oranı')
plt.legend()

plt.plot(epochs, loss, 'b', label='Eğitim Kaybı')
plt.plot(epochs, val_loss, 'r', label='Validation Kaybı')
plt.title('Model Kayıp Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp Oranı')
plt.legend()

plt.show()

#******************************************************************************

#Resimlerde maske takıp takmadıgını anlama
#Test resmimlerinin eklenmesi
images = ['test.jpg','test2.jpg','test3.jpg','test4.jpg','test5.jpg','test6.jpg','test7.jpg','test8.jpg','test9.jpg','test10.jpg','test11.jpg','test12.jpg']


eye_cascade=cv2.CascadeClassifier('haarcascade-eye.xml')

#Ornek path
img = images[9]   
    
img = plt.imread(img,format='8UC1')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
face_cascade=cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
faces = face_cascade.detectMultiScale(gray, 1.05, 10)

# Yüze dikdortgen cizme

for (x, y, w, h) in faces:

    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    (mask, withoutMask) = model.predict(face)[0]
    mask = mask*100
    withoutMask = withoutMask*100
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
     # Piksel boyutlarını bulma
    print("Image Width: " , w)
    textSize = cv2.getTextSize(text="Maske Yok: " + str("%.2f" % round(mask, 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)
    print("Text Width: " , textSize[0][0])
    
    if mask > withoutMask:
        cv2.putText(img,
                    text = "Maske Var: " + str("%.2f" % round(mask, 2)),
                    org = (x-5,y-15),
                    fontFace=font,
                    fontScale = (1.2*w)/textSize[0][0],
                    color = (0, 255, 0),
                    thickness = 3,
                    lineType = cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 5)
    else:
        cv2.putText(img,
                    text = "Maske Yok: " + str("%.2f" % round(withoutMask, 2)),
                    org = (x-5,y-15),
                    fontFace=font,
                    fontScale = (1.2*w)/textSize[0][0],
                    color = (255, 0, 0),
                    thickness = 3,
                    lineType = cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)

# Display
plt.imshow(img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imwrite("image1.jpeg",img)

#******************************************************************************
#Videoda bulma

cap = cv2.VideoCapture("MaskDetectionTestVideo.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter("MaskDetectionTestOutput.mp4",cv2.VideoWriter_fourcc(*'DIVX'),25,(width,height))

while True:
    
    ret, img = cap.read()
    
    if ret == True:
        time.sleep(1/25)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 8)

        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]
            mask = mask*100
            withoutMask = withoutMask*100

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Pikselleri bulma
            # print("Image Width: " , w)
            textSize = cv2.getTextSize(text="Maske Yok: " + str("%.2f" % round(mask, 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)
            # print("Text Width: " , textSize[0][0])
            if mask > withoutMask:
                cv2.putText(img,
                            text = "Maske Var: " + str("%.2f" % round(mask, 2)),
                            org = (x-5,y-20),
                            fontFace=font,
                            fontScale = (2*w)/textSize[0][0],
                            color = (0, 255, 0),
                            thickness = 3,
                            lineType = cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 5)
            else:
                cv2.putText(img,
                            text = "Maske Yok: " + str("%.2f" % round(withoutMask, 2)),
                            org = (x-5,y-20),
                            fontFace=font,
                            fontScale = (1.8*w)/textSize[0][0],
                            color = (0, 0, 255),
                            thickness = 3,
                            lineType = cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
        
        # Kaydetme
        writer.write(img)
                # Gosterme   
        cv2.imshow("MaskDetectionTest",img)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()


