import librosa as lib
import numpy as np

class Extract_Features:
    '''
    This class will extract many different features from an audio file into a numpy array.
    
    Reason : ML model will have many features to efficiently utilize supervised learning
    for sound classification
    
    Features to be extracted: (Based on article Giordano BL, McAdams S. Material
    identification of real impact sounds: Effects of size variation in steel)
    
    - Resonant Frequencies (the natural frequency something vibrates at when it is hit using Fourier Transform)
    - Spectral Centroid (the average frequency of the sound)
    - Temporal decay (Decay time of sound)
    - Attack Characteristics (inital impact sound)
    '''
    
    def __init__(self, sample_rate = 16000):
        # Same sample rate as AudioFileSource
        self.sample_rate = sample_rate
        
    def extract(self, audio):
        '''
        most important functions of this class. Takes audio numpy array and returns a dictionary with
        all the features listed above
        '''
            
        if audio is None:
            print("Please input a valid audio input")
            return None
        
        Features = {}

        #--------------------Attack Characteristics-------------------- 

        Features["resonant_frequency"] = self.get_resonant_frequency(audio)

        #------------------------Temporal decay------------------------

        Features["spectral_centroid"] = self.get_spectral_centroid(audio)

        #----------------------Resonant frequency----------------------

        Features["decay_rate"] = self.get_decay_rate(audio)

        #-----------------------Spectral envelope-----------------------

        Features["attack_strength"] = self.get_attack_strength(audio) 

        return Features

    def get_attack_strength(self, audio):
        # Takes the inital strength of the inital impact using the first 20ms of it.

        first_20 = int(0.02 * self.sample_rate)  
        first = audio[:first_20]

        # Check for empty audio if it goes through
        if len(first) == 0:
            return 0.0
        
        # Find the tallest wave with abs value
        attack_strength = np.max(np.abs(first))

        return float(attack_strength)
    
    def get_decay_rate(self, audio):
        # Measures the energy change in audio file from beginning to end

        # Root Mean Square gives a good measure of signal energy
        rms = lib.feature.rms(y=audio)[0]  # Takes square of every value, take avg then sqrt it.  Amount of times done varies on sample

        if len(rms) == 0:
            return 0.0

        # energy at the start and at the end
        initial_energy = rms[0]
        final_energy = rms[-1]

        # bigger difference means more decay
        decay_rate = initial_energy - final_energy

        return float(decay_rate)
    
    def get_resonant_frequency(self, audio):
        '''
        Finds the main resonant frequency in the sound.

        This uses the Fourier Transform to convert the 
        sound from the time domain into the frequency domain.

        Then it finds which frequency has the strongest peak.
        '''
        # apply FFT to the audio (into magnitudes)
        fft = np.abs(np.fft.rfft(audio))

        # create the matching frequency values
        frequencies = np.fft.rfftfreq(len(audio), d=(1 / self.sample_rate))

        # find the index of the highest frequency peak
        best_index = np.argmax(fft)

        # use that peak as the resonant frequency
        resonant_frequency = frequencies[best_index]

        return float(resonant_frequency)

    def get_spectral_centroid(self, audio):
        '''
        It tells us where most of the sound energy is located
        in the frequency spectrum.

        Higher centroid = brighter / sharper sound | something like class
        Lower centroid = darker / lower sound | something like wood
        '''
        centroid_values = lib.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]

        average_frequency = np.mean(centroid_values)

        return float(average_frequency)
    
    def get_feature_names(self):
        '''
        Returns the features in the exact order
        that should be used for training and prediction.
        '''

        return ["resonant_frequency", "spectral_centroid", "decay_rate", "attack_strength"]
    