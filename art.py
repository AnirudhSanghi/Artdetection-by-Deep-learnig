import tensorflow as tf
import os
import csv
import numpy as np
import pathlib
import cv2


checkpoint_path_1 = "training_temp(plain)/cp.ckpt"          #setting checkpoint path to acccess trained model later
checkpoint_path_2 = "training_main(plain)/cp.ckpt"          #where it was left
checkpoint_dir = os.path.dirname(checkpoint_path_1)


### accessing tarinig image from disk and converting them as a tensor quantity and labeling different classes.

train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    r'C:\Users\Anirudh Sanghi\PycharmProjects\Artdetector\CV\train_set', labels='inferred', label_mode='categorical', class_names=None,
    color_mode='rgb', batch_size=16, image_size=(100, 100), shuffle=True, seed=123,
    validation_split=0.3, subset= 'training', interpolation='bilinear', follow_links=False
)

### partioning given trainset to give 30% examples for dev set
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    r'C:\Users\Anirudh Sanghi\PycharmProjects\Artdetector\CV\train_set', labels='inferred', label_mode='categorical', class_names=None,
    color_mode='rgb', batch_size=16, image_size=(100, 100), shuffle=True, seed=123,
    validation_split=0.3, subset= 'validation', interpolation='bilinear', follow_links=False
)

classnames = train_ds.class_names                      # accessing all class names used in labeling strings to use later

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

def model_def(size=(100, 100, 3), classes=5):
    X_input = tf.keras.Input(size)          # creating a place holder for inputing image
    Normalising_data_set = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(X_input)  # normalising given rgb values in range of 0-255

    X = tf.keras.layers.ZeroPadding2D((3,3))(Normalising_data_set)                                   # zeropadding to give more weightage to corners and to tackle small dimension output

    X = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0001))(X)  # creating first convulation layer with padding-'valid'
    X = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(X)                                                                                 # maxpool of stride 2 and filter size 2x2
    X = tf.keras.layers.BatchNormalization(axis=3)(X)                                                                                  # batch normalisation to train better
    X = tf.keras.layers.Activation('relu')(X)                                                                                          # applying  activation using 'relu' fu


    X = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0001))(X)   #layer2
    X = tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0001))(X)  #layer3
    X = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0001))(X)  #layer4
    X = tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0001))(X)  ##layer5

    X = tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)


    X = tf.keras.layers.Flatten()(X)         # flattening the matrix input to feed dense layer

    X = tf.keras.layers.Dense(1024, activation='relu', name='fc' + str(1024_1), kernel_initializer='glorot_uniform',kernel_regularizer= tf.keras.regularizers.l2(0.0001))(X) #applying dense layer with relularisation effect
    X = tf.keras.layers.Dropout(0.5)(X)                                                                                                                                      #using drop out to dense layer to reduce no. of neurons with keep prob = 0.5
    X = tf.keras.layers.Dense(2048, activation='relu', name='fc' + str(1024_2), kernel_initializer='glorot_uniform',kernel_regularizer= tf.keras.regularizers.l2(0.0001))(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = 'glorot_uniform')(X)             # final output giving five classes prediction

    model = tf.keras.Model(inputs = X_input, outputs = X, name='model_def')

    return model


#### following code is used for checking best optimising model and it has been commented out

# def identity_block(X, f, filters, stage, block):
#     """
#     Implementation of the identity block as defined in Figure 4
#
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#
#     Returns:
#     X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
#     """
#
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     # Retrieve Filters
#     F1, F2, F3 = filters
#
#     # Save the input value. You'll need this later to add back to the main path.
#     X_shortcut = X
#
#     # First component of main path
#     X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
#                kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = tf.keras.layers.Activation('relu')(X)
#
#     ### START CODE HERE ###
#
#     # Second component of main path (≈3 lines)
#     X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
#                kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = tf.keras.layers.Activation('relu')(X)
#
#     # Third component of main path (≈2 lines)
#     X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
#                kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
#
#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = tf.keras.layers.Add()([X, X_shortcut])
#     X = tf.keras.layers.Activation('relu')(X)
#
#     ### END CODE HERE ###
#
#     return X
#
#
# def convolutional_block(X, f, filters, stage, block, s=2):
#     """
#     Implementation of the convolutional block as defined in Figure 4
#
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#     s -- Integer, specifying the stride to be used
#
#     Returns:
#     X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
#     """
#
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     # Retrieve Filters
#     F1, F2, F3 = filters
#
#     # Save the input value
#     X_shortcut = X
#
#     ##### MAIN PATH #####
#     # First component of main path
#     X = tf.keras.layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
#     X = tf.keras.layers.Activation('relu')(X)
#
#     ### START CODE HERE ###
#
#     # Second component of main path (≈3 lines)
#     X = tf.keras.layers.Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
#                kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
#     X = tf.keras.layers.Activation('relu')(X)
#
#     # Third component of main path (≈2 lines)
#     X = tf.keras.layers.Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
#
#     ##### SHORTCUT PATH #### (≈2 lines)
#     X_shortcut = tf.keras.layers.Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
#                         kernel_initializer='glorot_uniform')(X_shortcut)
#     X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
#
#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = tf.keras.layers.Add()([X,X_shortcut])
#     X = tf.keras.layers.Activation('relu')(X)
#
#     ### END CODE HERE ###
#
#     return X
#
#
# def ResNet50(input_shape=(64, 64, 3), classes=6):
#     """
#     Implementation of the popular ResNet50 the following architecture:
#     CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
#     -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
#
#     Arguments:
#     input_shape -- shape of the images of the dataset
#     classes -- integer, number of classes
#
#     Returns:
#     model -- a Model() instance in Keras
#     """
#
#     # Define the input as a tensor with shape input_shape
#     X_input = tf.keras.layers.Input(input_shape)
#
#     Normalising_data_set = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)(X_input)
#
#     # Zero-Padding
#     X = tf.keras.layers.ZeroPadding2D((3, 3))(Normalising_data_set)
#
#     # Stage 1
#     X = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), name='conv1', kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = tf.keras.layers.Activation('relu')(X)
#     X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
#
#     # Stage 2
#     X = convolutional_block(X, f=3, filters=[32, 32, 64], stage=2, block='a', s=1)
#     X = identity_block(X, 3, [32, 32, 64], stage=2, block='b')
#     X = identity_block(X, 3, [32, 32, 64], stage=2, block='c')
#
#     ### START CODE HERE ###
#
#     # Stage 3 (≈4 lines)
#     X = convolutional_block(X, f=3, filters=[64, 64, 128], stage=3, block='a', s=2)
#     X = identity_block(X, 4, [64, 64, 128], stage=3, block='b')
#     X = identity_block(X, 4, [64, 64, 128], stage=3, block='c')
#     X = identity_block(X, 4, [64, 64, 128], stage=3, block='d')
#
#     # Stage 4 (≈6 lines)
#     X = convolutional_block(X, f=3, filters=[128, 128, 256], stage=4, block='a', s=2)
#     X = identity_block(X, 5, [128, 128, 256], stage=4, block='b')
#     X = identity_block(X, 5, [128, 128, 256], stage=4, block='c')
#     X = identity_block(X, 5, [128, 128, 256], stage=4, block='d')
#     X = identity_block(X, 5, [128, 128, 256], stage=4, block='e')
#     # X = identity_block(X, 3, [128, 128, 256], stage=4, block='f')
#
#     # Stage 5 (≈3 lines)
#     X = convolutional_block(X, f=3, filters=[256, 256, 512], stage=5, block='a', s=2)
#     X = identity_block(X, 5, [256, 256, 512], stage=5, block='b')
#     X = identity_block(X, 5, [256, 256, 512], stage=5, block='c')
#
#     # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
#     X = tf.keras.layers.AveragePooling2D((2, 2))(X)
#
#     ### END CODE HERE ###
#
#     # output layer
#     X = tf.keras.layers.Flatten()(X)
#     X = tf.keras.layers.Dense(1024, activation='relu', name='fc' + str(1024_1), kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.Dense(2048, activation='relu', name='fc' + str(2048_2), kernel_initializer='glorot_uniform')(X)
#     X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer='glorot_uniform')(X)
#
#     # Create model
#     model = tf.keras.Model(inputs=X_input, outputs=X, name='ResNet50')
#
#     return model



model = model_def(size=(100, 100, 3), classes=5)

model.load_weights(checkpoint_path_1)             # loading weights from previously trained model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # setting up optimiser and loss category fro back propagation
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1,
                                                 save_weights_only=True,                 # calling checkpoint
                                                 verbose=1)

model.fit(train_ds, validation_data=test_ds ,epochs = 5,callbacks=[cp_callback])        # fitting the model to train the parameters



# print("Evaluate on test data")
# results = model.evaluate(test_ds, batch_size=128)                                      # for evaluating on previosly trained model(personal clearification)
# print("test loss, test acc:", results)

csv_path= r"C:\Users\Anirudh Sanghi\PycharmProjects\Artdetector\CV\submission.csv"
location_gen = r"C:\Users\Anirudh Sanghi\PycharmProjects\Artdetector\CV\test_set\{}"

base_value = 0.5                  # setting up the  base value for each class as a threshold to be confirm that test image belong to that class
with open(csv_path, mode='r') as csv_file, \
         open('submission.csv', 'w', newline='') as write_obj:           # opening csv file to read image name row by row and also to writw in new csv file
    csv_reader = csv.reader(csv_file)
    csv_writer = csv.writer(write_obj)
    line_count = 0


    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            row.append('label')
            csv_writer.writerow(row)
            line_count += 1
        else:
            name = row[0]                            #extracting name of the image to be test or label  from csv file
            print(name)
            fin = location_gen.format(name)          #concatinating the image name with test folder path to search the image in test folder
            path = pathlib.Path(fin)                 # giving  a path to the image extracted from a particular row to feed be feed tp prediction model
            print(path)
            image= cv2.imread(fin)                   # raeding the imaze file
            image = tf.image.resize(image,(100,100)) # resizing the test image to a fixed same dim. beacuse test folder contain different dimension of each image
            image = np.expand_dims(image,axis=0)     # adding 1 more axis to he image tensor as model input feed also have batch dim.(here 1)
            prediction = model.predict(image)        # finally predicting all classes(5) probability (all btw 0 and 1) and return a 1x5 probability array
            print(prediction.shape)
            stor = 7                                 # random initilisation of a variable


            for i in range(len(classnames)):         #code for ensuring that a class to be finalise
                if prediction[0][i] > base_value:    # to have probability greater than base value
                    stor = i
                    row.append(classnames[i])        #appending the row extracted from readable csv file with classname that it has been predicted
                    csv_writer.writerow(row)         #filling the appended row in the new writable csv file
                    line_count+=1

                    break

            if stor != 10:                           # when we able to predict a class overcoming the threshold(base_value)
                print('\n')
                print(classnames[stor])
                print('\n')
            else:                                    # when no class's probability is greater than the base_value
                row.append(' ')                      #so appending that row with whitespace characters to show unable to predict
                csv_writer.writerow(row)

            print(classnames)
            print(prediction)
            print('---------------------------------------------')


    print(f'Processed {line_count} lines.')
   


#### Note-- above code has been designed as per my pc like path varibles for traing set,test set,checkpoint etc
#### So, it would not run ,to run it modify path variables as per individual pc

