import numpy as np
import pandas as pd
import librosa

def get_tempo(y, sr):
    tempo = librosa.feature.tempo(y=y, sr=sr)
    return tempo

def get_short_time_energy(y, hop_length=256, frame_length=256):
    ste = np.array([
        sum(abs(y[j:j+frame_length] ** 2) / frame_length)
        for j in range(0, len(y), hop_length)
    ])
    ste_feature = np.hstack([ste.mean(), ste.std()])
    
    return ste_feature

def get_root_mean_square_energy(y, hop_length=512, frame_length=512):
    rms = librosa.feature.rms(y=y)
    rms_feature = np.hstack([rms.mean(), rms.std()])
    
    return rms_feature

def get_zcr(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feature = np.hstack([zcr.mean(), zcr.std()])
    
    return zcr_feature

def get_spectral_bandwidth(y, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_feature = np.hstack([bandwidth.mean(), bandwidth.std()])
    
    return bandwidth_feature

def get_spectral_contrast(y, sr):
    stft = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = contrast.mean(axis=1)
    contrast_std = contrast.std(axis=1)
    contrast_feature = np.hstack([contrast_mean, contrast_std])
    
    return contrast_feature

def get_mfcc(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
    
    return mfcc_feature

def get_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_std = chroma.std(axis=1)
    chroma_feature = np.hstack([chroma_mean, chroma_std])
    
    return chroma_feature

def extract_features(file):
    n_samples = 5

    feature_labels = ['Tempo', 'STE_mean', 'STE_std', 'RMS_mean', 'RMS_std', 'ZCR_mean', 'ZCR_std',
                        'Bandwidth_mean', 'Bandwidth_std', 'Contrast0_mean', 'Contrast1_mean', 'Contrast2_mean',
                        'Contrast3_mean', 'Contrast4_mean', 'Contrast5_mean', 'Contrast6_mean', 'Contrast0_std',
                        'Contrast1_std', 'Contrast2_std', 'Contrast3_std', 'Contrast4_std', 'Contrast5_std',
                        'Contrast6_std', 'MFCC0_mean', 'MFCC1_mean', 'MFCC2_mean', 'MFCC3_mean', 'MFCC4_mean',
                        'MFCC5_mean', 'MFCC6_mean', 'MFCC7_mean', 'MFCC8_mean', 'MFCC9_mean', 'MFCC10_mean',
                        'MFCC11_mean', 'MFCC12_mean', 'MFCC13_mean', 'MFCC14_mean', 'MFCC15_mean', 'MFCC16_mean',
                        'MFCC17_mean', 'MFCC18_mean', 'MFCC19_mean', 'MFCC0_std', 'MFCC1_std', 'MFCC2_std', 'MFCC3_std',
                        'MFCC4_std', 'MFCC5_std', 'MFCC6_std', 'MFCC7_std', 'MFCC8_std', 'MFCC9_std', 'MFCC10_std',
                        'MFCC11_std', 'MFCC12_std', 'MFCC13_std', 'MFCC14_std', 'MFCC15_std', 'MFCC16_std', 'MFCC17_std',
                        'MFCC18_std', 'MFCC19_std', 'Chroma0_mean', 'Chroma1_mean', 'Chroma2_mean', 'Chroma3_mean',
                        'Chroma4_mean', 'Chroma5_mean', 'Chroma6_mean', 'Chroma7_mean', 'Chroma8_mean', 'Chroma9_mean',
                        'Chroma10_mean', 'Chroma11_mean', 'Chroma0_std', 'Chroma1_std', 'Chroma2_std', 'Chroma3_std',
                        'Chroma4_std', 'Chroma5_std', 'Chroma6_std', 'Chroma7_std', 'Chroma8_std', 'Chroma9_std',
                        'Chroma10_std', 'Chroma11_std']
    feature_matrix = np.zeros((n_samples, len(feature_labels)))

    features_to_drop = ['Bandwidth_mean', 'Contrast2_mean', 'Contrast3_mean', 'MFCC11_std', 'MFCC13_std', 'MFCC15_std',
                        'MFCC17_std', 'MFCC18_std','STE_std', 'MFCC0_mean', 'RMS_mean']

    y, sr = librosa.load(file)
    y_trim, _ = librosa.effects.trim(y, top_db=40)

    sample_length_seconds = 20
    sample_length = int(sample_length_seconds * sr)
    total_length = y_trim.shape[0]

    start = 0
    end = sample_length
    i_sample = 0

    while (end < total_length) & (i_sample < n_samples):
        sample = y_trim[start:end]

        tempo = get_tempo(sample, sr)
        ste = get_short_time_energy(sample)
        rms = get_root_mean_square_energy(sample)
        zcr = get_zcr(sample)
        bandwidth = get_spectral_bandwidth(sample, sr)
        contrast = get_spectral_contrast(sample, sr)
        mfcc = get_mfcc(sample, sr)
        chroma = get_chroma(sample, sr)

        feature_vector = np.hstack([tempo, ste, rms, zcr, bandwidth, contrast, mfcc, chroma])

        feature_matrix[i_sample] = feature_vector

        start = end
        end += sample_length
        i_sample += 1

    df = pd.DataFrame(feature_matrix, columns=feature_labels)
    df = df.drop(features_to_drop, axis=1)
    return df