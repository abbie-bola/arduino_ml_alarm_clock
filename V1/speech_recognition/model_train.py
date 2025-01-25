import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import os 

DATA_PATH = "alarm_data.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
SAVED_MODEL_PATH = "alarm_model.keras"
BATCH_SIZE = 32
NUM_KEYWORDS = 74 # number of keywords in dataset
DATASET_PATH = "C:\\Users\\phoen\\Documents\\Code\\AI_ML\\Tensorflow_DeepLearning_Tutorial\\Arduino_Alarm\\Voice_Recordings_for_AI\\"

def load_dataset(data_path): 
    """loads dataset"""
    with open(data_path, "r") as fp: 
        data = json.load(fp)

    # extract inputs and targets 
    X = data["MFCCs"]
    y = data["labels"]

    # they are currently python lists
    # Convert X and y to numpy arrays
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    
    return X, y


def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    """This function splits the dataset into sets for training, validation and testing the model
    Args:
        data_path(string or constant): location of the dataset to be split
        test_size(float): proportion of dataset to be used for testing
                    default value = 0.1 -> 10% of the dataset will be used for testing purposes
                       
        test_validation: proportion of training set to be used for validation purposes
                    default value = 0.1 -> 10% of training set (90%) to be used for validation.
                                    therefore, 9% of overall dataset to be used for validating the model.
    """
    # load dataset
    X, y = load_dataset(data_path)

    # create train/validation/test splits 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=test_validation)

    # convert input from 2d to 3d array  
    # current shape [# of segments, 13] 

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# def representative_data_gen():
"""Accepts no args. Function for generating representaive data for int-8 quantization"""
#     # X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)
#     # #rep_test = tf.convert_to_tensor(X_validation, dtype=tf.float32)
#     # #input_shape = (1, X_validation.shape[1], X_validation.shape[2], X_validation.shape[3])
#     # rep_test = X_validation[np.newaxis, ...]
#     with open(DATA_PATH, "r") as fp: 
#         data = json.load(fp)
        
#     model_keys = ['mappings', 'labels', 'MFCCs']
#     for i in range(300):
#         rep = {key:data[key][i] for key, value in data.items() if key in model_keys}
#         return rep
#     a = rep.items()
#     b = list(a)
#     n_rep = np.array(b) 
#     yield n_rep
    

def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"): 
    """Builds Convolutional Neural Network (CNN), to carry out machine learning of the dataset
    Args:
        input_shape (array)
        learning_rate (float)
        error (string): error/loss function
                    default value: "sparse_categorical_crossentropy" 

    NOTE: During inference, model only accepts audio data > 1 second. 
    Otherwise, it will return error, due to wrong input shape provided for Dense layer than comes after Flattened layer
                    
    """

    # build network 
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # convolutional layer 1
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu",
                                              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    
    # convolutional layer 2 
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu",
                                              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    
    # convolutional layer 3 
    model.add(keras.layers.Conv2D(32, (2,2), activation="relu",
                                              kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    
    # flatten the output and feed it into dense layer (because Dense layers require 1D array as input)
    model.add(keras.layers.Flatten()
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax")) # outputs array of scores (probabilities) for predictions. e.g [0.1, 0.7, 0.2, ...]
    
    # compile the model 
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    # print model overview 
    model.summary()

    return model


def main():

    # load the train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN model 
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # creates a 3d array, because this CNN requires 3D input. [# of segments, # 13 coefficients, depth=1]  
    model = build_model(input_shape, LEARNING_RATE)
    
    # train the model 
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(X_validation, y_validation))
    
    # evaluate the model 
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, Test accuracy: {test_accuracy}")

    # save the model 
    model.save(SAVED_MODEL_PATH)

    
    """For int-8 Quantization"""
    # # Load your trained model
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)  
    
    # # Set quantization options - 8-bit quant (not used here)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_data_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8  # or tf.uint8 if required
    # converter.inference_output_type = tf.int8
    
    # # Convert and save the quantized model
    # quantized_model = converter.convert()
    # open('alarm_quant_model.tflite', 'wb').write(quantized_model)

    """For Default Quantization"""
    # convert model to tflite - default quant

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    open('alarm_lite_model.tflite', 'wb').write(tflite_model)
        

if __name__ == "__main__":
    main()
