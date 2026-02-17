import numpy as np
import librosa as lb

class AudioFileSource:
    '''Loads an audio source from a file into a numpy array'''
    
    def __init__(self, sample_rate = 16000):
        self.sample_rate = sample_rate

    def load(self, file_path):
        #returns a numpy array
        try:
            audio, sample_rate = librosa.load(file_path, sr=self.sample_rate)
            return audio
        
        except Exception as reason:
            #handles invalid files
            print(f"Please import a valid audio file (WAV, MP3, FLAC, OGG, AIFF, M4A)")
            print(e)
