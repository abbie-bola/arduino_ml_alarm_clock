import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "replace_with_model_path"
NUM_SAMPLES_TO_CONSIDER = 22050 # 1 second worth of sound

model = keras.models.load_model(
    MODEL_PATH, custom_objects=None, compile=True, safe_mode=True
)

class _Keyword_Spotting_Service: 
    # this is a singleton class to perform inference on the Speech Recognition CNN
    
    model = None 
    mappings = [
        "addie",
        "alarm",
        "eight",
        "eighteen",
        "eleven",
        "fifteen",
        "fifty",
        "fifty_eight",
        "fifty_five",
        "fifty_four",
        "fifty_nine",
        "fifty_one",
        "fifty_seven",
        "fifty_six",
        "fifty_three",
        "fifty_two",
        "five",
        "forty",
        "forty_eight",
        "forty_five",
        "forty_four",
        "forty_nine",
        "forty_one",
        "forty_seven",
        "forty_six",
        "forty_three",
        "forty_two",
        "four",
        "fourteen",
        "hey",
        "hey addie",
        "nine",
        "nineteen",
        "no",
        "off",
        "on",
        "one",
        "set",
        "set_alarm",
        "set_timer",
        "seven",
        "seventeen",
        "six",
        "sixteen",
        "sixty",
        "stop",
        "ten",
        "thirteen",
        "thirty",
        "thirty_eight",
        "thirty_five",
        "thirty_four",
        "thirty_nine",
        "thirty_one",
        "thirty_seven",
        "thirty_six",
        "thirty_three",
        "thirty_two",
        "three",
        "timer",
        "twelve",
        "twenty",
        "twenty_eight",
        "twenty_five",
        "twenty_four",
        "twenty_nine",
        "twenty_one",
        "twenty_seven",
        "twenty_six",
        "twenty_three",
        "twenty_two",
        "two",
        "yes",
        "zero"
    ]

    _instance = None # keyword spotting instance

    
    def predict(self, file_path): 

        # extract the MFCCs 
        MFCCs = self.preprocess(file_path) # shape -> (# of segments, # of coefficients)

        # convert 2D MFCCs array into 4D array -> (# of samples, # of segments, # of coefficients, # of channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
    
        # make prediction -> creates an array of the probabilities of predicted mappings
        predictions = self.model.predict(MFCCs) # [[0.1, 0.6, 0.1, ...]]
        predicted_index = np.argmax(predictions) # returns index with highest probability
        predicted_keyword = self.mappings[predicted_index] # returns mappings value of predicted index

        return predicted_keyword
        
    
    def preprocess (self, file_path, n_mfcc=13, n_fft=2048, hop_length=512): 

        # load the audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER: 
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]
            
        # extract the MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft,
                                     hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service(): 

    # ensure that there is only 1 instance of Keyword_Spotting_Service
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__": 

    kss = Keyword_Spotting_Service()

    keyword_1 = kss.predict("replace with audio path")


    print(f"Predicted keywords: {keyword_1}")