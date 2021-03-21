from flask import Flask, render_template, request,redirect, url_for
import os
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.preprocessing import image


def modeling():
    input_dir = os.path.join(r'C:\Users\Lahiru\Desktop\Dataset\kvasir-dataset-v2')
    output_dir = os.path.join(r'C:\Users\Lahiru\Desktop\Dataset\kvasir-dataset-v2_splitted')

    splitfolders.ratio(input_dir, output=output_dir, seed=1337, ratio=(.7, .3), group_prefix=None)

    train_dir = os.path.join(r'C:\Users\Lahiru\Desktop\Dataset\kvasir-dataset-v2_splitted\train')
    test_dir = os.path.join(r'C:\Users\Lahiru\Desktop\Dataset\kvasir-dataset-v2_splitted\val')
    
    train_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_dir,
                                  target_size = (75,75),
                                  batch_size = 214,
                                  class_mode = 'categorical',
                                  subset='training')
 
    
    test_generator = test_datagen.flow_from_directory(test_dir,
                                 target_size=(75,75),
                                 batch_size = 37,
                                 class_mode = 'categorical')    
        
    classes=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'] 

    X_train=[]
    for j in range(len(train_generator)):    
        for m in train_generator[j][0]: 
            X_train.append(m)
        
    y_train=[]
    for i in range(len(train_generator)):
        for k in train_generator[i][1]:  
            y_train.append(np.argmax(k))    

    X_test=[]        
    for j in range(len(test_generator)):    
        for m in test_generator[j][0]: 
            X_test.append(m)

    y_test=[]
    for i in range(len(test_generator)):
        for k in test_generator[i][1]:  
            y_test.append(np.argmax(k))
           
    X_train=np.array(X_train)
    y_train=np.array(y_train)

    X_test=np.array(X_test)
    y_test=np.array(y_test)


    cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='softmax')
    ])  

    cnn.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = cnn.fit(X_train, y_train, epochs=16)


    return cnn
cnn = modeling()




app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():       
        result = ["zero"]
        return render_template("Home.html",result=result)  
    
    
       


@app.route('/Predict', methods=['POST', 'GET'])
def predict():              
        img_url = request.form['nm1']
      
        classes=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'] 
        pic_dir = img_url
        l = pic_dir.split("\\")
        print(pic_dir)
        test_image = image.load_img(pic_dir, target_size = (75, 75))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        res = cnn.predict(test_image)

        y_classes = np.argmax(res)
        correcting = l[7]
        predicting = classes[y_classes]
        result = []       
        result.append(correcting)
        result.append(predicting)
     
        print(result)
      
        return render_template("Home.html",result=result)  
    


if __name__ == '__main__':          
    app.run(debug = True)