import numpy as np 
import cv2 
import os 
from sklearn.model_selection import train_test_split 
import seaborn as sns 
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator
import pickle # modeli kayıt etmek için 

path = "SonDataSet" 
myList = os.listdir(path) 
numberOfClasses = len(myList) # clas sayısı 6 -> [1,2,3,4,5,6]
print(numberOfClasses)
images = []
ClassNo = []

for i in range(numberOfClasses):
    myImageList = os.listdir(path + "\\" + str(i)) # [0,1,2,3,4,5,6,7,8,9] dosya yolları
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j) # resimlerin dosya yolları 
        img = cv2.resize(img, (224, 224))
        images.append(img) 
        ClassNo.append(i)
        
print("Datalar Yuklendi")

images = np.array(images)
ClassNo = np.array(ClassNo)

x_train, x_test, y_train, y_test = train_test_split(images, ClassNo, test_size = 0.2, random_state = 42 )

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img 

x_train = np.array(list(map(preProcess, x_train))) # preprocess fonksiyonunu bütün x_train verilerine uygular 
x_test = np.array(list(map(preProcess, x_test)))

x_train = x_train.reshape(-1, 224, 224, 1) # (-1) veri boyutunu otomatik ayarla (32,32,1) x,y,channel
x_test = x_test.reshape(-1, 224, 224, 1)

#data generate 
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    rotation_range=10
    )

dataGen.fit(x_train)

y_train = to_categorical(y_train, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)  

# model 
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(224, 224, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

batch_size = 64 

model.fit(x_train, y_train, epochs=5, batch_size=batch_size, verbose=2, validation_data=(x_test, y_test))
# modelin çıktısnı görselleştirme 
#hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_validation, y_validation),epochs=15, steps_per_epoch= x_train.shape[0]//batch_size,shuffle=1)

# modeli depolamak 
pickle_out = open("model_trained_hand_new.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

        