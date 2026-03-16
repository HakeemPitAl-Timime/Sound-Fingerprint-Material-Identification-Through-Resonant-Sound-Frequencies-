import librosa as lib
import numpy as np

class Extract_Features:
    '''
    Extract the required features from a 1D numpy audio array.
    Features returned (in this order):
      - resonant_frequency (Hz)
      - spectral_centroid (Hz)
      - decay_rate (1/seconds)  (estimated time constant)
      - attack_strength (relative RMS of initial attack)
    '''

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract(self, audio):
        """
        Given a 1D numpy array `audio` (mono), compute and return
        a dict with the four features.
        """
        # ensure numpy array
        audio = np.asarray(audio, dtype=float)

        # Trim leading/trailing silence to make calculations more robust
        audio, _ = lib.effects.trim(audio, top_db=40)

        # resonant_frequency: find peak in magnitude spectrum
        resonant = self._resonant_frequency(audio)

        # spectral centroid (librosa returns array over frames; take mean)
        spectral_centroid = self._spectral_centroid(audio)

        # decay_rate: estimate exponential decay time constant from envelope
        decay = self._decay_rate(audio)

        # attack_strength: ratio of initial RMS energy to median RMS energy
        attack = self._attack_strength(audio)

        return {
            "resonant_frequency": float(resonant),
            "spectral_centroid": float(spectral_centroid),
            "decay_rate": float(decay),
            "attack_strength": float(attack)
        }

    def _resonant_frequency(self, audio):
        # Compute magnitude spectrum using FFT
        n = len(audio)
        if n < 2:
            return 0.0
        # apply a window to reduce spectral leakage
        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        # ignore DC and very low frequencies (<20 Hz)
        valid = freqs > 20
        if not np.any(valid):
            return 0.0

        idx = np.argmax(spectrum[valid])
        # map back to index in full array
        resonant_freq = freqs[valid][idx]
        return resonant_freq

    def _spectral_centroid(self, audio):
        # librosa returns centroid per frame; take mean
        try:
            centroid_values = lib.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            return float(np.mean(centroid_values))
        except Exception:
            # fallback: compute centroid from FFT
            n = len(audio)
            if n < 2:
                return 0.0
            spectrum = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
            if spectrum.sum() == 0:
                return 0.0
            return float(np.sum(freqs * spectrum) / np.sum(spectrum))

    def _decay_rate(self, audio):
        """
        Estimate decay using an envelope approach.
        We compute the RMS envelope, take its log, and fit a linear slope.
        The slope (negative) approximates exponential decay: envelope ~ exp(k * t),
        so decay_rate = -slope (in 1/seconds). If fit fails, return 0.0.
        """
        # parameters for RMS
        hop_length = max(1, int(self.sample_rate * 0.01))  # 10 ms
        frame_length = max(2, int(self.sample_rate * 0.02))  # 20 ms

        try:
            rms = lib.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            times = np.arange(len(rms)) * (hop_length / self.sample_rate)
            # avoid zeros
            rms = np.maximum(rms, 1e-8)
            log_rms = np.log(rms)

            # find peak (attack) index and fit after the peak
            peak_idx = int(np.argmax(rms))
            if peak_idx + 3 >= len(log_rms):
                return 0.0
            x = times[peak_idx+1:]
            y = log_rms[peak_idx+1:]
            if len(x) < 3:
                return 0.0
            # linear fit
            A = np.vstack([x, np.ones_like(x)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            decay_rate = -m  # since log(rms) ~ -decay_rate * t + c
            if decay_rate < 0:
                decay_rate = 0.0
            return float(decay_rate)
        except Exception:
            return 0.0

    def _attack_strength(self, audio):
        """
        Measure the relative energy in the initial attack window (e.g., first 30 ms)
        compared to the median RMS of the whole clip.
        Returns a value >= 0 where higher means stronger attack.
        """
        total_len = len(audio)
        window_ms = 30
        window_samples = max(1, int(self.sample_rate * window_ms / 1000.0))
        # compute RMS across whole clip and initial window
        try:
            full_rms = np.mean(lib.feature.rms(y=audio)[0]) + 1e-8
            init = audio[:window_samples]
            init_rms = np.mean(lib.feature.rms(y=init)[0]) + 1e-8
            return float(init_rms / full_rms)
        except Exception:
            # fallback: use simple energy ratio
            init_energy = np.sum(audio[:window_samples]**2) + 1e-8
            full_energy = np.sum(audio**2) / max(1, total_len) + 1e-8
            return float((init_energy / window_samples) / full_energy)

    def get_feature_names(self):
        '''
        Returns the features in the exact order
        that should be used for training and prediction.
        '''
        return ["resonant_frequency", "spectral_centroid", "decay_rate", "attack_strength"]
