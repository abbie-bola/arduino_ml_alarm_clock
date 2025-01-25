
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, AirAbsorption, TimeMask,  BandPassFilter
# HighPassFilter requires pydub 
import librosa
import soundfile as sf
import os
DATASET_PATH = "C:\\Users\\phoen\\Documents\\Code\\AI_ML\\Tensorflow_DeepLearning Tutorial\\Voice_Recordings_for_AI\\"

def augment_audio(dataset_path):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.4),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.001),
        HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=4000, p=0.3),
        BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=0.5),
        TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.6),
        AirAbsorption(min_distance=10.0, max_distance=50.0, p=0.2)
    ])
    
    # loop through all the sub-directories 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # at each iteration, obtains directory, sub-directory and filenames in each folder
        # we need to ensure that we're not at root level 
        if dirpath is not dataset_path: 

            # loop through all the file names and extract MFCCs 
            for f in filenames:
        
                # get file path
                file_path = os.path.join(dirpath, f) # concatenates dirpath and filename
                
                # # load audio file 
                # signal, sr = librosa.load(file_path)
        
                signal, sr = librosa.load(file_path)
                augmented_signal = augment(signal, sr)
                
                # save augmented_signal
                augmented_file = sf.write("aug_{}.wav".format(f), augmented_signal, sr) 
                
        

if __name__ == "__main__":
    augment_audio(DATASET_PATH)
