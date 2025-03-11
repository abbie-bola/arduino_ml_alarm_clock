import tensorflow as tf
import keras
import tensorflow_io as tfio
import os
import pathlib
import numpy as np
import librosa
import noisereduce as nr
import matplotlib as plt
from sklearn.model_selection import train_test_split


# Define dataset path
DATASET_PATH = "{path to your local dataset}"
data_dir = pathlib.Path(DATASET_PATH)

# Define parameters
SAMPLING_RATE = 16000
OUTPUT_LENGTH = 16000

##### FUNCTIONS #####
def load_and_resample_audio(file_path):
    """Function to load and resample audio to 16kHz (if needed)
    Parameters: 
        file_path: variable/str 
        A variable or string containing the path to local dataset location (folder)
    Returns:
        y: np.ndarray [shape=(n,) or (…, n)]
        audio time series. Multi-channel is supported.

        sr: number = 16000 [scalar]
        sampling rate of y
    """
    
    # Load audio file with librosa
    audio, sr = librosa.load(file_path, sr=SAMPLING_RATE)
    
    # Resample if needed
    if sr != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    # Ensure audio is the correct length (1 sec)
    if len(audio) > OUTPUT_LENGTH:
        audio = audio[:OUTPUT_LENGTH]
    else:
        audio = np.pad(audio, (0, OUTPUT_LENGTH - len(audio)))
    
    return audio

# Preprocessing function to reduce noise in audio samples
def reduce_noise(audio, label):
    # Nested function to handle noise_reduce numpy operations
    def noise_reduction(audio_np): 
        sr = 16000
        return nr.reduce_noise(y=audio_np, sr=sr, prop_decrease=0.7)

    #tf.py_function to wrap the noise reduction function and convert output to a tensor
    audio = tf.py_function(noise_reduction, [audio], tf.float32)
    
    # Ensure the output has the shape (16000,)
    audio.set_shape([16000])

    return audio, label

# Add channel dimension to audio sample. changes shape from (...,) to (..., channels)
def expand_dimension(audio, label):
    audio = tf.expand_dims(audio, -1)
    return audio, label

# Create a dataset of audio files and their labels
def generate_data(data_dir, labels):
    """Function to generate data arrays from contents of dataset directory
    Parameters: 
        data_dir: variable
        A variable containing the PurePath of DATASET_PATH (local dataset location)
        labels: dict
        A dictionary containing keyword names derived from DATASET_PATH subfolders
    Returns:
        audio_train_tensor, label_train_tensor, audio_val_tensor, label_val_tensor, audio_test_tensor, label_test_tensor:
        
        Tensors of preprocessed audio files and their corresponding labels, which will make up the train, validation and test datasets
    """
    audio_files = []
    audio_labels = []

    print("Preprocessing and Generating Data")
    # iterates through the label dict
    for label in labels:
        label_dir = data_dir / label # concatenates label name to root folder -> creates subfolder directory for that specific label e.g '/home/V4audio' / 'cat' -> /home/V4audio/cat 
        for audio_file in label_dir.iterdir(): # iterates through each audio file in label_dir 
            if audio_file.suffix == '.wav': # if audio is .wav file, load, resample audio; append resampled audio and corresponding integer labels into their respective lists
                audio = load_and_resample_audio(str(audio_file))
                audio_files.append(audio) # append audio array to list
                audio_labels.append(label_to_index[label])  # Append index of corresponding label to list
    print("Audio Loaded and Resampled, Labels Converted to Integers.")
    
    # Convert lists to NumPy arrays
    audio_files = np.array(audio_files)
    audio_labels = np.array(audio_labels, dtype=np.int32)  # Ensure labels are integers
    
    print("Starting Split 1")
    # Perform stratified split (shuffling while preserving class distribution)
    audio_train, audio_val, label_train, label_val = train_test_split(
        audio_files, 
        audio_labels, 
        test_size=0.2,  # For example, 80% for training, 20% for validation
        stratify=audio_labels,  # This ensures stratified splitting
        random_state=42
    )

    print("Starting Split 2")
    # Perform stratified split for validation and test data (further split the validation set)
    audio_val, audio_test, label_val, label_test = train_test_split(
        audio_val,
        label_val,
        test_size=0.5,  # 50% of the validation set will go to test (effectively splitting into 10% test, 10% val)
        stratify=label_val,  # Ensure stratified splitting
        random_state=42
    )

    print("Converting to Tensors")
    # Convert sets to TensorFlow tensors
    audio_train_tensor = tf.convert_to_tensor(audio_train, dtype=tf.float32)
    label_train_tensor = tf.convert_to_tensor(label_train, dtype=tf.int32)

    audio_val_tensor = tf.convert_to_tensor(audio_val, dtype=tf.float32)
    label_val_tensor = tf.convert_to_tensor(label_val, dtype=tf.int32)

    audio_test_tensor = tf.convert_to_tensor(audio_test, dtype=tf.float32)
    label_test_tensor = tf.convert_to_tensor(label_test, dtype=tf.int32)
    print("Tensor Creation Successful!")
    
    return audio_train_tensor, label_train_tensor, audio_val_tensor, label_val_tensor, audio_test_tensor, label_test_tensor

def generate_dataset(audio_train_tensor, label_train_tensor, audio_val_tensor, label_val_tensor, audio_test_tensor, label_test_tensor):
    """Function to tf.data.Dataset from audio and label tensors
    Parameters: 
        audio_train_tensor, label_train_tensor, audio_val_tensor, label_val_tensor, audio_test_tensor, label_test_tensor:
        Tensors created from train/test split of audio and label arrays

    Returns:
        train_dataset, val_dataset, test_dataset:
        tf.data.Dataset objects for training, validation and testing model
    """
    print("Creating Datasets")

    # Create datasets for train, validation, and test with channel dimension
    train_dataset = tf.data.Dataset.from_tensor_slices((audio_train_tensor, label_train_tensor))  # Add channel dimension
    val_dataset = tf.data.Dataset.from_tensor_slices((audio_val_tensor, label_val_tensor))  # Add channel dimension
    test_dataset = tf.data.Dataset.from_tensor_slices((audio_test_tensor, label_test_tensor))  # Add channel dimension

    print("Datasets Created Successfully")
    
    return train_dataset, val_dataset, test_dataset


##### IMPLEMENTATION #####
# Get the folder names (labels)
labels = [label.name for label in data_dir.iterdir() if label.is_dir()]

# Convert label names to a dictionary {label_name: index}
label_to_index = {label: index for index, label in enumerate(labels)}
print("Label Mapping:", label_to_index)

# Generate the data tensors
audio_train_tensor, label_train_tensor, audio_val_tensor, label_val_tensor, audio_test_tensor, label_test_tensor = generate_data(data_dir, labels)

# Generate tf.data.Dataset objects
train_dataset, val_dataset, test_dataset = generate_dataset(audio_train_tensor, label_train_tensor, audio_val_tensor, label_val_tensor, audio_test_tensor, label_test_tensor)
print("Datasets generated successfully! Shuffling Datasets.")

# Shuffle the train dataset with a buffer size of 5000
train_dataset = train_dataset.shuffle(5000, reshuffle_each_iteration=True)
# Shuffle the validation and test datasets (optional, can be adjusted based on your needs)
val_dataset = val_dataset.shuffle(5000, reshuffle_each_iteration=True)
test_dataset = test_dataset.shuffle(5000, reshuffle_each_iteration=True)
print("Datasets shuffled successfully!")

# Preview the datasets; recommend taking 15-30 items to confirm it's shuffled properly
for audio, label in train_dataset.take(1):
    print(f'Train Audio shape: {audio.shape}, Label shape: {label.shape}')

for audio, label in val_dataset.take(1):
    print(f'Val Audio shape: {audio.shape}, Label shape: {label.shape}')

for audio, label in test_dataset.take(1):
    print(f'Test Audio shape: {audio.shape}, Label shape: {label.shape}')

# view the specification of the dataset elements, i.e shape, dtype, name
print("Train Dataset Specifications (Original):", train_dataset.element_spec)
print("Validation Dataset Specifications (Original):", val_dataset.element_spec)
print("Test Dataset Specifications (Original):", test_dataset.element_spec)

# perform transformations on datasets
train_dataset = train_dataset.map(reduce_noise, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.map(expand_dimension, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(expand_dimension, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(expand_dimension, num_parallel_calls=tf.data.AUTOTUNE)

# view the new specification of the dataset elements, i.e shape, dtype, name
print("Train Dataset Specifications (New):", train_dataset.element_spec)
print("Validation Dataset Specifications (New):", val_dataset.element_spec)
print("Test Dataset Specifications (New):", test_dataset.element_spec)

# Batch the datasets
BATCH_SIZE = 64
train_dataset = train_dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # no shuffle to ensure model is tested on the same dataset at all times




