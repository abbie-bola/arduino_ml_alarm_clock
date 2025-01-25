import librosa 
import os
import json 


DATASET_PATH = "C:\\Users\\phoen\\Documents\\Code\\AI_ML\\Tensorflow_DeepLearning Tutorial\\Arduino_Alarm\\Voice_Recordings_for_AI\\"
JSON_PATH = "alarm_data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 second worth of sound (sample rate = 22050) -> when loading audio file with librosa, there are 22050 audio samples in a second

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """This function reads all the audio files,
    extracts the mfcc, and stores them in a json file
    Params: 
        dataset_path (string): path of audio dataset to be processed
        json_path (string): path where processed & labelled data is stored
        n_mfcc (int): number of MFCCs  
        hop_length (int) 
        n_fft (int): number of fast fourier transforms
    """
    # data dictionary
    data = {
        "mappings": [],
        "labels": [], # expected outputs
        "MFCCs": [], # inputs
        "files": [] # saves analysed/processed audio files in path (helps for later analysis)
    }

    # loop through all the sub-directories 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # at each iteration, obtains directory, sub-directory and filenames in each folder
        # we need to ensure that we're not at root level 
        if dirpath is not dataset_path: 

            # update mappings
            category = dirpath.split("\\")[-1] # splits pathname. Output: dataset/down -> [dataset, down]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through all the file names and extract MFCCs 
            for f in filenames:

                # get file path
                file_path = os.path.join(dirpath, f) # concatenates dirpath and filename

                # load audio file 
                signal, sr = librosa.load(file_path)
                
                # ensure the audio file is at least 1 second
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # enforce 1-sec-long signal
                    signal = signal[:SAMPLES_TO_CONSIDER] # obtains signals between 0 and 22050 

                    # extract the MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc,
                                                 hop_length=hop_length, n_fft=n_fft)

                    # store data 
                    data["labels"].append(i-1) # stores subdirectories in list. (i-1) is because currently, the root directory is part of the indexes, so that needs to change.
                                            # e.g S_Subset = 0, down = 1, up = 2 -> down = 0, up = 1
                    data["MFCCs"].append(MFCCs.T.tolist()) # MFCC is currently an ndarray. This line casts it into a list, to enable it to be stored in a JSON file 
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}") # prints file_path and corresponding label


    # store in JSON file 
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        

if __name__ == "__main__": 
    prepare_dataset(DATASET_PATH, JSON_PATH)