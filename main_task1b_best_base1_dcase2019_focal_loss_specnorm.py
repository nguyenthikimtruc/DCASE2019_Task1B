#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:40:04 2019

@author: truc

Based on https://github.com/McDonnell-Lab/DCASE2019-Task1

Run on terminal with command line to save a log file:
    
python main.py >>main_base12.log 2>main_base12.err


"""


# coding: utf-8

# In[ ]:


# select a GPU
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[ ]:


#imports 
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import librosa
import librosa.display
import soundfile as sound
import pickle
import copy
import dill


import keras
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_data_format('channels_last')

import tensorflow
from keras.optimizers import SGD, Adadelta
from keras.callbacks import ModelCheckpoint
from keras.utils import CustomObjectScope

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from DCASE2019_models_5 import model_best2019_base1, model_best2019_base2,create_model_db_3cnn_nexp

from DCASE_training_functions import LR_WarmRestart

from DCASE_plots import plot_confusion_matrix

from focal_loss import categorical_focal_loss

from mixup_generator import MixupGenerator

from DenseMoE import DenseMoE

import random
random.seed(30)

print ("Random number with seed 30") 
print("Librosa version = ",librosa.__version__)
print("Pysoundfile version = ",sound.__version__)
print("keras version = ",keras.__version__)
print("tensorflow version = ",tensorflow.__version__)


# In[ ]:


#WhichTask = '1a'
WhichTask = '1b'
#WhichTask = '1c'

if WhichTask =='1a':
    DatasetPath = '../TAU-urban-acoustic-scenes-2019-development/'
    TrainFile = DatasetPath + 'evaluation_setup/fold1_train.csv'
    ValFile = DatasetPath + 'evaluation_setup/fold1_evaluate.csv'
    sr = 48000
    num_audio_channels = 2
elif WhichTask =='1b':
    DatasetPath = '/srv/TUG/truc/DCASE2019/task1/datasets/TAU-urban-acoustic-scenes-2019-mobile-development/'
    TrainFile = DatasetPath + 'evaluation_setup/fold1_train.csv'
    ValFile = DatasetPath + 'evaluation_setup/fold1_evaluate.csv'
    sr = 44100
    num_audio_channels = 1
elif WhichTask =='1c':
    DatasetPath = '../Task1c/'
    TrainFile = DatasetPath + 'evaluation_setup/fold1_train.csv'
    sr = 44100
    num_audio_channels = 1
    
SampleDuration = 10

#log-mel spectrogram parameters
NumFreqBins = 128 # 256
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2) #4)
NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))
eps = np.spacing(1) 

#training parameters
max_lr = 0.5
batch_size = 64
num_epochs = 510
mixup_alpha = 0.2



# flags 
normalization_flag = True  ##False  ##


# In[]: Manage model filename

model_names = [
#        'model_best2019_base1_SpeCor_refBC_specNorm_mel_BNaxis2_focalloss.h5',
#        'model_best2019_base1_SpeCor_refBC_specNorm_logmel_BNaxis2_focalloss.h5',
#        'model_best2019_base1_SpeCor_refABC_specNorm_mel_BNaxis2_focalloss.h5',
#        'model_best2019_base1_SpeCor_refABC_specNorm_logmel_BNaxis2_focalloss.h5',
#        'model_best2019_base1_NoSpecCor_specNorm_mel_BNaxis2_focalloss.h5',
#        'model_best2019_base1_NoSpecCor_specNorm_logmel_BNaxis2_focalloss.h5',
#        
        'model_db3cnn_10MoE_SpeCor_refBC_specNorm_mel_BNaxis1_focalloss.h5',
        'model_db3cnn_10MoE_SpeCor_refBC_specNorm_logmel_BNaxis1_focalloss.h5',
        
        'model_db3cnn_10MoE_NoSpecCor_specNorm_mel_BNaxis1_focalloss.h5',
        'model_db3cnn_10MoE_NoSpecCor_specNorm_logmel_BNaxis1_focalloss.h5',       
        
  ]


#def main():
# In[]: Manage path for storing system outputs
Basepath = os.getcwd() + '/' 
spectrum_path = Basepath + 'system_outputs/task1b_128_431/spectrums/'
if not os.path.exists(spectrum_path):
    os.makedirs(spectrum_path)
    
refBC_logmel_feature_path = Basepath + 'system_outputs/task1b_128_431/logmel_features/' #refBC_features_logmel/'  # for logmel_features
if not os.path.exists(refBC_logmel_feature_path):
    os.makedirs(refBC_logmel_feature_path)
    
refBC_mel_feature_path = Basepath + 'system_outputs/task1b_128_431/mel_features/'  # for mel_features
if not os.path.exists(refBC_mel_feature_path):
    os.makedirs(refBC_mel_feature_path)
    
refABC_logmel_feature_path = Basepath + 'system_outputs/task1b_128_431/refABC_features_logmel/'  # for logmel_features
if not os.path.exists(refABC_logmel_feature_path):
    os.makedirs(refABC_logmel_feature_path)
    
refABC_mel_feature_path = Basepath + 'system_outputs/task1b_128_431/refABC_features_mel/'  # for mel_features
if not os.path.exists(refABC_mel_feature_path):
    os.makedirs(refABC_mel_feature_path)
    
NoSpecCor_logmel_feature_path = Basepath + 'system_outputs/task1b_128_431/NoSpecCor_features_logmel/'  # for logmel_features
if not os.path.exists(NoSpecCor_logmel_feature_path):
    os.makedirs(NoSpecCor_logmel_feature_path)
    
NoSpecCor_mel_feature_path = Basepath + 'system_outputs/task1b_128_431/NoSpecCor_features_mel/'  # for mel_features
if not os.path.exists(NoSpecCor_mel_feature_path):
    os.makedirs(NoSpecCor_mel_feature_path)
    
normalization_path = Basepath + 'system_outputs/task1b_128_431/normalizations_check/'
if not os.path.exists(normalization_path):
    os.makedirs(normalization_path)
    
model_path = Basepath + 'system_outputs/task1b_128_431/learners/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = Basepath + 'system_outputs/task1b_128_431/recognizers/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
log_path = Basepath + 'log_128_431/'
if not os.path.exists(log_path):
    os.makedirs(log_path)   
    
  
# In[ ]:Set loging file

# Log title
print('DCASE2019 / Task1B -- Acoustic scene classification')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++')

print('saved to log_128_431/db3cnn_10MoE_refBCSpeCor_specNorm_BNaxis1_focalloss.log')
print('saved to system_outputs_check')

# In[ ]: Manage labels and training and validation sets

'''There are 16530 files in audio but 1030 files unsed for training set
   number of audio files for training set and val set is 15500
'''

#load filenames and labels
dev_train_df = pd.read_csv(TrainFile,sep='\t', encoding='ASCII')
dev_val_df = pd.read_csv(ValFile,sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
wavpaths_val = dev_val_df['filename'].tolist()
wavpaths_all = sorted(wavpaths_train + wavpaths_val)

filename_list_train = [os.path.splitext(os.path.basename(filename))[0] for filename in wavpaths_train]
filename_list_val = [os.path.splitext(os.path.basename(filename))[0] for filename in wavpaths_val]
filename_list = sorted(filename_list_train + filename_list_val)

y_train_labels =  dev_train_df['scene_label'].astype('category').cat.codes.values
y_val_labels =  dev_val_df['scene_label'].astype('category').cat.codes.values

ClassNames = np.unique(dev_train_df['scene_label'])
NumClasses = len(ClassNames)

y_train = keras.utils.to_categorical(y_train_labels, NumClasses)
y_val = keras.utils.to_categorical(y_val_labels, NumClasses)


# In[ ]:Do spectrum extraction


# Feature extraction stage
print('Spectrum Extraction')
print('========================================')
if not len(wavpaths_all)==len(filename_list): 
    print('Error!!!')


#spectrum = np.zeros((NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list):
    
    audio_filename = DatasetPath + wavpaths_all[i]
    spectrum_filename = spectrum_path + filename + '.cpickle'
    #
    if not os.path.isfile(spectrum_filename):
        print('{}/{} {}.wav'.format(i, len(filename_list), filename))
        # Read sound file
        stereo,fs = sound.read(audio_filename,stop=SampleDuration*sr)
        for channel in range(num_audio_channels):
            if len(stereo.shape)==1:
                stereo = np.expand_dims(stereo,-1)
            # Do mel extraction
            spectrum = np.abs(librosa.core.stft(stereo[:,channel]+ eps,
                                           n_fft      = NumFFTPoints,
                                           hop_length = HopLength,
                                           window     = 'hann', 
                                           center     = True)
            )
                
        ##save feature file
        pickle.dump(spectrum, open(spectrum_filename, "wb" ) )

print('-----DONE') 

       
        
# In[ ]:Calculated spectrum correction coefficients    
        

print('Spectrum Correction using refBC')
print('========================================')
print('-----Calculated spectrum correction coefficients using refBC')  

refBC_spectrumcorrection_filename_path = normalization_path + 'refBC_spectrumcorrection.cpickle'

if not os.path.isfile(refBC_spectrumcorrection_filename_path):  
    ## Processing spectrum correction
    filename_list_a = []
    filename_list_b = []
    filename_list_c = []
    same_filename_list_b = []
    
    # Get filename list of dev B -- preparing spectrum pairs for spectrum correction 
    for filename in filename_list_train:
        if filename[-2:] == '-b':
            # Get feature filename
            spectrum_filename = spectrum_path + filename + '.cpickle'
            filename_list_b.append(spectrum_filename)
            # Get same string in feature filename
            same_filename_list_b.append(filename[:-2])
                    
    # Get filename for dev.A and C which have same filenames of Dev.B    
    for filename in filename_list_train:
        for same_filename_b in same_filename_list_b:
            # Get the same filename for dev C
            if filename[-2:] == '-c' and filename[:-2] == same_filename_b:
                spectrum_filename = spectrum_path + filename + '.cpickle'
                filename_list_c.append(spectrum_filename)
            # Get the same filename for Dev A    
            if filename[-2:] == '-a' and filename[:-2]==same_filename_b:
                spectrum_filename = spectrum_path + filename + '.cpickle'
                filename_list_a.append(spectrum_filename)   
                    
    # Do spectrum correction for data_ref = Dev. A and data_x = Dev. B / C   
    sum_divided_spectrum = None    
    sum_divided_spectrum_ref = None 
    sum_divided_spectrum_ref_1 = None 
    for spectrum_a, spectrum_b, spectrum_c,  in zip(filename_list_a, filename_list_b, filename_list_c):
        # Load feature matrix
        features_a = pickle.load(open(spectrum_a, 'rb'))
        features_b = pickle.load(open(spectrum_b, 'rb'))
        features_c = pickle.load(open(spectrum_c, 'rb'))
        
        # Accumulate of divided spectrum statistics
        data_ref1=features_b
        data_ref2=features_c
        data_x=features_a
        #
        data_ref = np.mean((data_ref1 + data_ref2)/2., axis=1)
        data_ref_ref = (np.mean(data_ref1, axis=1) + np.mean(data_ref2, axis=1))/2.
        data_ref_ref_1 = (data_ref1 + data_ref2)/2.
        divided_spectrum = data_ref/np.mean(data_x, axis=1)
        divided_spectrum_ref = data_ref_ref/np.mean(data_x, axis=1)
        #divided_spectrum_ref_1 = data_ref_ref_1/data_x 
        '''dividing first in for loop and finally doing mean will cause the coefficient in difference
        divided_spectrum_ref_1 = data_ref_ref_1/data_x
    #coefficients_ref_1 = (np.mean((sum_divided_spectrum_ref_1/len(filename_list_a)),axis=1)).reshape(-1, 1)
    #coefficients_ref_2 = (np.mean(sum_divided_spectrum_ref_1,axis=1)/len(filename_list_a)).reshape(-1, 1)
        '''
        #
        if sum_divided_spectrum is None:
            sum_divided_spectrum = divided_spectrum
            sum_divided_spectrum_ref = divided_spectrum_ref
            #sum_divided_spectrum_ref_1 = divided_spectrum_ref_1
        else:
            sum_divided_spectrum += divided_spectrum
            sum_divided_spectrum_ref += divided_spectrum_ref
            #sum_divided_spectrum_ref_1 += divided_spectrum_ref_1
        #
    # Calculate coefficients of spectrum correction for all spectrum pairs
    coefficients = (sum_divided_spectrum/len(filename_list_a)).reshape(-1, 1)
    coefficients_ref = (sum_divided_spectrum_ref/len(filename_list_a)).reshape(-1, 1)
    #coefficients_ref_1 = (np.mean((sum_divided_spectrum_ref_1/len(filename_list_a)),axis=1)).reshape(-1, 1)
    #coefficients_ref_2 = (np.mean(sum_divided_spectrum_ref_1,axis=1)/len(filename_list_a)).reshape(-1, 1)
    if (np.abs(coefficients - coefficients_ref)).all() < 0.00001:
        print('Same coefficients')
    ##save feature file
    pickle.dump(coefficients, open(refBC_spectrumcorrection_filename_path, 'wb' ) )
         
print('-----DONE') 
       
# In[ ]:Calculated spectrum correction coefficients using reference of 3 Devices   
        

print('Spectrum Correction using refABC')
print('========================================')
print('-----Calculated spectrum correction coefficients using refABC')  

refABC_spectrumcorrection_filename_path = normalization_path + 'refABC_spectrumcorrection.cpickle'

if not os.path.isfile(refABC_spectrumcorrection_filename_path):  
    ## Processing spectrum correction
    filename_list_a = []
    filename_list_b = []
    filename_list_c = []
    same_filename_list_b = []
    
    # Get filename list of dev B -- preparing spectrum pairs for spectrum correction 
    for filename in filename_list_train:
        if filename[-2:] == '-b':
            # Get feature filename
            spectrum_filename = spectrum_path + filename + '.cpickle'
            filename_list_b.append(spectrum_filename)
            # Get same string in feature filename
            same_filename_list_b.append(filename[:-2])
                    
    # Get filename for dev.A and C which have same filenames of Dev.B    
    for filename in filename_list_train:
        for same_filename_b in same_filename_list_b:
            # Get the same filename for dev C
            if filename[-2:] == '-c' and filename[:-2] == same_filename_b:
                spectrum_filename = spectrum_path + filename + '.cpickle'
                filename_list_c.append(spectrum_filename)
            # Get the same filename for Dev A    
            if filename[-2:] == '-a' and filename[:-2]==same_filename_b:
                spectrum_filename = spectrum_path + filename + '.cpickle'
                filename_list_a.append(spectrum_filename)   
                    
    # Do spectrum correction for data_ref = Dev. A and data_x = Dev. B / C   
    sum_divided_spectrum = None    
    sum_divided_spectrum_ref = None 
    sum_divided_spectrum_ref_1 = None 
    for spectrum_a, spectrum_b, spectrum_c,  in zip(filename_list_a, filename_list_b, filename_list_c):
        # Load feature matrix
        features_a = pickle.load(open(spectrum_a, 'rb'))
        features_b = pickle.load(open(spectrum_b, 'rb'))
        features_c = pickle.load(open(spectrum_c, 'rb'))
        
        # Accumulate of divided spectrum statistics
        data_ref1=features_b
        data_ref2=features_c
        data_ref3=features_a
        data_x=features_a
        #
        data_ref = np.mean((data_ref1 + data_ref2 + data_ref2)/3., axis=1)
        data_ref_ref = (np.mean(data_ref1, axis=1) + np.mean(data_ref2, axis=1) + np.mean(data_ref3, axis=1))/3.
        data_ref_ref_1 = (data_ref1 + data_ref2 + data_ref2)/3.
        divided_spectrum = data_ref/np.mean(data_x, axis=1)
        divided_spectrum_ref = data_ref_ref/np.mean(data_x, axis=1)
        #divided_spectrum_ref_1 = data_ref_ref_1/data_x 
        '''dividing first in for loop and finally doing mean will cause the coefficient in difference
        divided_spectrum_ref_1 = data_ref_ref_1/data_x
    #coefficients_ref_1 = (np.mean((sum_divided_spectrum_ref_1/len(filename_list_a)),axis=1)).reshape(-1, 1)
    #coefficients_ref_2 = (np.mean(sum_divided_spectrum_ref_1,axis=1)/len(filename_list_a)).reshape(-1, 1)
        '''
        #
        if sum_divided_spectrum is None:
            sum_divided_spectrum = divided_spectrum
            sum_divided_spectrum_ref = divided_spectrum_ref
            #sum_divided_spectrum_ref_1 = divided_spectrum_ref_1
        else:
            sum_divided_spectrum += divided_spectrum
            sum_divided_spectrum_ref += divided_spectrum_ref
            #sum_divided_spectrum_ref_1 += divided_spectrum_ref_1
        #
    # Calculate coefficients of spectrum correction for all spectrum pairs
    coefficients = (sum_divided_spectrum/len(filename_list_a)).reshape(-1, 1)
    coefficients_ref = (sum_divided_spectrum_ref/len(filename_list_a)).reshape(-1, 1)
    #coefficients_ref_1 = (np.mean((sum_divided_spectrum_ref_1/len(filename_list_a)),axis=1)).reshape(-1, 1)
    #coefficients_ref_2 = (np.mean(sum_divided_spectrum_ref_1,axis=1)/len(filename_list_a)).reshape(-1, 1)
    if (np.abs(coefficients - coefficients_ref)).all() < 0.00001:
        print('Same coefficients')
    ##save feature file
    pickle.dump(coefficients, open(refABC_spectrumcorrection_filename_path, 'wb' ) )
         
print('-----DONE') 

# In[ ]:Do feature extraction


#load spectrums and spectrum coefficients and 
print('Feature Extraction using reference Device B and C')
print('========================================')
print('-----Do feature extraction using reference Device B and C')

refBC_SC_coefficients = pickle.load(open(refBC_spectrumcorrection_filename_path, 'rb'))
print('refBC_SC_coefficients =')
print(refBC_SC_coefficients)

#mel_feature = np.zeros((NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list):
    
    spectrum_filename = spectrum_path + filename + '.cpickle'
    refBC_mel_feature_filename = refBC_mel_feature_path + filename + '.cpickle'
    refBC_logmel_feature_filename = refBC_logmel_feature_path + filename + '.cpickle'
    #
    if not (os.path.isfile(refBC_mel_feature_filename)\
            and os.path.isfile(refBC_logmel_feature_filename)):
        print('{}/{} {}.wav'.format(i,len(filename_list),filename))  
        # Read spectrum coefficients
        spectrum = pickle.load(open(spectrum_filename, 'rb'))
#        spectrum = copy.deepcopy(spectrum)
        refBC_corrected_spectrum = spectrum * refBC_SC_coefficients
        
        ## display spectrum
#        import matplotlib.pyplot as plt
#        plt.figure()
#        librosa.display.specshow(corrected_spectrum, y_axis='mel', x_axis='time')
#        plt.colorbar(format='%+2.0f dB')
#        plt.title('Spectrum')
#        plt.tight_layout()
#        plt.show()
            
        # mel basis   mel_basis.shape=(128, 1025)
        mel_basis = librosa.filters.mel(sr    = sr,
                                       n_fft  = NumFFTPoints,
                                       n_mels = NumFreqBins,
                                       fmin   = 0.0,
                                       fmax   = sr/2,
                                       htk    = True,
                                       norm   = None)
                                     
        refBC_mel_feature = np.dot(mel_basis, refBC_corrected_spectrum)
        refBC_logmel_feature = np.log(np.abs(refBC_mel_feature) + eps)
        #---mel_feature.shape = (128, 431)
        
#        ## display spectrogram
#        import matplotlib.pyplot as plt
#        plt.figure()
#        librosa.display.specshow(log_mel_feature, y_axis='mel', x_axis='time')
#        plt.colorbar(format='%+2.0f dB')
#        plt.title('Spectrum')
#        plt.tight_layout()
#        plt.show()

        ##save feature file
        pickle.dump(refBC_mel_feature, open(refBC_mel_feature_filename, 'wb' ) )
        pickle.dump(refBC_logmel_feature, open(refBC_logmel_feature_filename, 'wb' ) )
    
print('-----DONE')    

# In[ ]:Do feature extraction using reference of 3 Devices


#load spectrums and spectrum coefficients and 
print('Feature Extraction mel')
print('========================================')
print('-----Do feature extraction using reference of 3 Devices')

refABC_SC_coefficients = pickle.load(open(refABC_spectrumcorrection_filename_path, 'rb'))
print('refABC_SC_coefficients =')
print(refABC_SC_coefficients)

#mel_feature = np.zeros((NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list):
    
    spectrum_filename = spectrum_path + filename + '.cpickle'
    refABC_mel_feature_filename = refABC_mel_feature_path + filename + '.cpickle'
    refABC_logmel_feature_filename = refABC_logmel_feature_path + filename + '.cpickle'
    #
    if not os.path.isfile(refABC_mel_feature_filename):
        print('{}/{} {}.wav'.format(i,len(filename_list),filename))  
        # Read spectrum coefficients
        spectrum = pickle.load(open(spectrum_filename, 'rb'))
#        spectrum = copy.deepcopy(spectrum)
        refABC_corrected_spectrum = spectrum * refABC_SC_coefficients
            
        # mel basis   mel_basis.shape=(128, 1025)
        mel_basis = librosa.filters.mel(sr    = sr,
                                       n_fft  = NumFFTPoints,
                                       n_mels = NumFreqBins,
                                       fmin   = 0.0,
                                       fmax   = sr/2,
                                       htk    = True,
                                       norm   = None)
                                     
        refABC_mel_feature = np.dot(mel_basis, refABC_corrected_spectrum)
        refABC_log_mel_feature = np.log(np.abs(refABC_mel_feature) + eps)
        ##save feature file
        pickle.dump(refABC_mel_feature, open(refABC_mel_feature_filename, 'wb' ) )
        pickle.dump(refABC_log_mel_feature, open(refABC_logmel_feature_filename, 'wb' ) )
print('-----DONE')   


# In[ ]:Do zero mean and unit variance normalization for feature size (128, 431)
'''If feature size has more channel than 1 (128, 431,1)
   normalization more complecated
   This code use for 1 channel 
'''

print('-----Calculated zero mean and unit variance Normalization')  

refBC_logmel_normalization_filename_path = normalization_path + 'refBC_normalization_logmel.cpickle'

refBC_mel_normalization_filename_path = normalization_path + 'refBC_normalization_mel.cpickle'


if not (os.path.isfile(refBC_logmel_normalization_filename_path) and \
        os.path.isfile(refBC_mel_normalization_filename_path)):
          
    print('Calculated zero mean and unit variance Normalization for logmel')  

    
    refBC_logmel_features_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins),'float32')
    for i, filename in enumerate(filename_list_train):
        refBC_logmel_feature_filename = refBC_logmel_feature_path + filename + '.cpickle'
        # load feature
        refBC_logmel_features = pickle.load(open(refABC_logmel_feature_filename, 'rb'))  
        refBC_logmel_features_train[i,:,:] = refBC_logmel_features[:,:]
        
        
    refBC_logmel_feature_mean_finalize = np.mean(refBC_logmel_features_train, axis=(0,2)).reshape(-1,1)
    # Finalize features_std_accummulate
    refBC_logmel_feature_std_finalize = np.std(refBC_logmel_features_train, axis=(0,2)).reshape(-1,1)   
    
    refBC_logmel_stats = {
            'mean': refBC_logmel_feature_mean_finalize,
            'std': refBC_logmel_feature_std_finalize
            }   
    
    print('Calculated zero mean and unit variance Normalization for mel ')
    refBC_mel_features_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins),'float32')
    for i, filename in enumerate(filename_list_train):
        refBC_mel_feature_filename = refBC_mel_feature_path + filename + '.cpickle'
        # load feature
        refBC_mel_features = pickle.load(open(refBC_mel_feature_filename, 'rb'))  
        refBC_mel_features_train[i,:,:] = refBC_mel_features[:,:]
        
        
    refBC_mel_feature_mean_finalize = np.mean(refBC_mel_features_train, axis=(0,2)).reshape(-1,1)
    # Finalize features_std_accummulate
    refBC_mel_feature_std_finalize = np.std(refBC_mel_features_train, axis=(0,2)).reshape(-1,1)
    
    refBC_mel_stats = {'mean': refBC_mel_feature_mean_finalize,
             'std': refBC_mel_feature_std_finalize
            }   
     
    ##save feature file
    pickle.dump(refBC_logmel_stats, open(refBC_logmel_normalization_filename_path, 'wb' ) )
    pickle.dump(refBC_mel_stats, open(refBC_mel_normalization_filename_path, 'wb' ) )
    
    print('refBC_feature_logmel ----------------')
    print('min = {} '.format(np.min(refBC_logmel_features_train)))
    print('max = {} '.format(np.max(refBC_logmel_features_train)))
    print() 
    
    print('refBC_feature_mean_finalize ')
    print('min = {} '.format(np.min(refBC_logmel_feature_mean_finalize)))
    print('max = {} '.format(np.max(refBC_logmel_feature_mean_finalize)))
    print()
    
    print('refBC_feature_std_finalize')
    print('min = {} '.format(np.min(refBC_logmel_feature_std_finalize)))
    print('max = {} '.format(np.max(refBC_logmel_feature_std_finalize)))
    print()
    
    print('refBC_feature_mel------------------ ')
    print('min = {} '.format(np.min(refBC_mel_features_train)))
    print('max = {} '.format(np.max(refBC_mel_features_train)))
    print() 
    
    print('refBC_feature_mean_finalize ')
    print('min = {} '.format(np.min(refBC_mel_feature_mean_finalize)))
    print('max = {} '.format(np.max(refBC_mel_feature_mean_finalize)))
    print()
    
    print('refBC_feature_std_finalize ')
    print('min = {} '.format(np.min(refBC_mel_feature_std_finalize)))
    print('max = {} '.format(np.max(refBC_mel_feature_std_finalize)))
    print() 
    
print('-----DONE') 


# In[ ]:Do zero mean and unit variance normalization for using reference of 3 Devices
'''If feature size has more channel than 1 (128, 431,1)
   normalization more complecated
   This code use for 1 channel 
'''

print('-----Calculated zero mean and unit variance Normalization')  

refABC_logmel_normalization_filename_path = normalization_path + 'refABC_normalization_logmel.cpickle'

refABC_mel_normalization_filename_path = normalization_path + 'refABC_normalization_mel.cpickle'


if not (os.path.isfile(refABC_logmel_normalization_filename_path) and \
        os.path.isfile(refABC_mel_normalization_filename_path)):
          
    print('Calculated zero mean and unit variance Normalization for logmel')  

    
    refABC_logmel_features_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins),'float32')
    for i, filename in enumerate(filename_list_train):
        refABC_logmel_feature_filename = refABC_logmel_feature_path + filename + '.cpickle'
        # load feature
        refABC_logmel_features = pickle.load(open(refABC_logmel_feature_filename, 'rb'))  
        refABC_logmel_features_train[i,:,:] = refABC_logmel_features[:,:]
        
        
    refABC_logmel_feature_mean_finalize = np.mean(refABC_logmel_features_train, axis=(0,2)).reshape(-1,1)
    # Finalize features_std_accummulate
    refABC_logmel_feature_std_finalize = np.std(refABC_logmel_features_train, axis=(0,2)).reshape(-1,1)   
    
    refABC_logmel_stats = {
            'mean': refABC_logmel_feature_mean_finalize,
            'std': refABC_logmel_feature_std_finalize
            }   
    
    print('Calculated zero mean and unit variance Normalization for mel ')
    refABC_mel_features_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins),'float32')
    for i, filename in enumerate(filename_list_train):
        refABC_mel_feature_filename = refABC_mel_feature_path + filename + '.cpickle'
        # load feature
        refABC_mel_features = pickle.load(open(refABC_mel_feature_filename, 'rb'))  
        refABC_mel_features_train[i,:,:] = refABC_mel_features[:,:]
        
        
    refABC_mel_feature_mean_finalize = np.mean(refABC_mel_features_train, axis=(0,2)).reshape(-1,1)
    # Finalize features_std_accummulate
    refABC_mel_feature_std_finalize = np.std(refABC_mel_features_train, axis=(0,2)).reshape(-1,1)
    
    refABC_mel_stats = {'mean': refABC_mel_feature_mean_finalize,
             'std': refABC_mel_feature_std_finalize
            }   
     
    ##save feature file
    pickle.dump(refABC_logmel_stats, open(refABC_logmel_normalization_filename_path, 'wb' ) )
    pickle.dump(refABC_mel_stats, open(refABC_mel_normalization_filename_path, 'wb' ) )
    
    print('refABC_feature_logmel ----------------')
    print('min = {} '.format(np.min(refABC_logmel_features_train)))
    print('max = {} '.format(np.max(refABC_logmel_features_train)))
    print() 
    
    print('refABC_feature_mean_finalize ')
    print('min = {} '.format(np.min(refABC_logmel_feature_mean_finalize)))
    print('max = {} '.format(np.max(refABC_logmel_feature_mean_finalize)))
    print()
    
    print('refABC_feature_std_finalize')
    print('min = {} '.format(np.min(refABC_logmel_feature_std_finalize)))
    print('max = {} '.format(np.max(refABC_logmel_feature_std_finalize)))
    print()
    
    print('refABC_feature_mel------------------ ')
    print('min = {} '.format(np.min(refABC_mel_features_train)))
    print('max = {} '.format(np.max(refABC_mel_features_train)))
    print() 
    
    print('refABC_feature_mean_finalize ')
    print('min = {} '.format(np.min(refABC_mel_feature_mean_finalize)))
    print('max = {} '.format(np.max(refABC_mel_feature_mean_finalize)))
    print()
    
    print('refABC_feature_std_finalize ')
    print('min = {} '.format(np.min(refABC_mel_feature_std_finalize)))
    print('max = {} '.format(np.max(refABC_mel_feature_std_finalize)))
    print() 
    
print('-----DONE') 


# In[ ]:Do feature extraction without spectrum correction


#load spectrums and spectrum coefficients and 
print('Feature Extraction without spectrum correction from audio file')
print('========================================')
print('-----Do feature extraction without spectrum correction' )


#spectrum = np.zeros((NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list):
    
    audio_filename = DatasetPath + wavpaths_all[i]
    
    NoSpecCor_mel_feature_filename = NoSpecCor_mel_feature_path + filename + '.cpickle'
    NoSpecCor_logmel_feature_filename = NoSpecCor_logmel_feature_path + filename + '.cpickle'
    
    #
    if not (os.path.isfile(NoSpecCor_mel_feature_filename)
            and os.path.isfile(NoSpecCor_logmel_feature_filename)
            ):
        print('{}/{} {}.wav'.format(i, len(filename_list), filename))
        # Read sound file
        stereo,fs = sound.read(audio_filename,stop=SampleDuration*sr)
        for channel in range(num_audio_channels):
            if len(stereo.shape)==1:
                stereo = np.expand_dims(stereo,-1)
            # Do mel extraction
            spectrum = np.abs(librosa.core.stft(stereo[:,channel]+ eps,
                                           n_fft      = NumFFTPoints,
                                           hop_length = HopLength,
                                           window     = 'hann', 
                                           center     = True)
            )
        

        mel_basis = librosa.filters.mel(sr    = sr,
                                       n_fft  = NumFFTPoints,
                                       n_mels = NumFreqBins,
                                       fmin   = 0.0,
                                       fmax   = sr/2,
                                       htk    = True,
                                       norm   = None)
                                     
        NoSpecCor_mel_feature = np.dot(mel_basis, spectrum)
        NoSpecCor_logmel_feature = np.log(np.abs(NoSpecCor_mel_feature) + eps)

        ##save feature file
        pickle.dump(NoSpecCor_mel_feature, open(NoSpecCor_mel_feature_filename, 'wb' ) )
        pickle.dump(NoSpecCor_logmel_feature, open(NoSpecCor_logmel_feature_filename, 'wb' ) )
    
print('-----DONE')   


# In[ ]:Do feature extraction without spectrum correction

#
##load spectrums and spectrum coefficients and 
#print('Feature Extraction without spectrum correction')
#print('========================================')
#print('-----Do feature extraction without spectrum correction' )
#
#
##mel_feature = np.zeros((NumFreqBins,NumTimeBins,num_audio_channels),'float32')
#for i, filename in enumerate(filename_list):
#    
#
#    NoSpecCor_mel_feature_filename = NoSpecCor_mel_feature_path + filename + '.cpickle'
#    NoSpecCor_logmel_feature_filename = NoSpecCor_logmel_feature_path + filename + '.cpickle'
#    #
#    if not (os.path.isfile(NoSpecCor_mel_feature_filename)\
#            and os.path.isfile(NoSpecCor_logmel_feature_filename)):
#        print('{}/{} {}.wav'.format(i,len(filename_list),filename))  
#        # Read spectrum coefficients
#        spectrum = pickle.load(open(spectrum_filename, 'rb'))
#
#        mel_basis = librosa.filters.mel(sr    = sr,
#                                       n_fft  = NumFFTPoints,
#                                       n_mels = NumFreqBins,
#                                       fmin   = 0.0,
#                                       fmax   = sr/2,
#                                       htk    = True,
#                                       norm   = None)
#                                     
#        NoSpecCor_mel_feature = np.dot(mel_basis, spectrum)
#        NoSpecCor_logmel_feature = np.log(np.abs(NoSpecCor_mel_feature) + eps)
#
#        ##save feature file
#        pickle.dump(NoSpecCor_mel_feature, open(NoSpecCor_mel_feature_filename, 'wb' ) )
#        pickle.dump(NoSpecCor_logmel_feature, open(NoSpecCor_logmel_feature_filename, 'wb' ) )
#    
#print('-----DONE')   


# In[ ]:Do zero mean and unit variance normalization without Spectrum correction
'''If feature size has more channel than 1 (128, 431,1)
   normalization more complecated
   This code use for 1 channel 
'''

print('-----Calculated zero mean and unit variance Normalization')  

NoSpecCor_logmel_normalization_filename_path = normalization_path + 'NoSpecCor_normalization_logmel.cpickle'

NoSpecCor_mel_normalization_filename_path = normalization_path + 'NoSpecCor_normalization_mel.cpickle'


if not (os.path.isfile(NoSpecCor_logmel_normalization_filename_path) and \
        os.path.isfile(NoSpecCor_mel_normalization_filename_path)):
          
    print('Calculated zero mean and unit variance Normalization for logmel')  

    
    NoSpecCor_logmel_features_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins),'float32')
    for i, filename in enumerate(filename_list_train):
        NoSpecCor_logmel_feature_filename = NoSpecCor_logmel_feature_path + filename + '.cpickle'
        # load feature
        NoSpecCor_logmel_features = pickle.load(open(NoSpecCor_logmel_feature_filename, 'rb'))  
        NoSpecCor_logmel_features_train[i,:,:] = NoSpecCor_logmel_features[:,:]
        
        
    NoSpecCor_logmel_feature_mean_finalize = np.mean(NoSpecCor_logmel_features_train, axis=(0,2)).reshape(-1,1)
    # Finalize features_std_accummulate
    NoSpecCor_logmel_feature_std_finalize = np.std(NoSpecCor_logmel_features_train, axis=(0,2)).reshape(-1,1)   
    
    NoSpecCor_logmel_stats = {
            'mean': NoSpecCor_logmel_feature_mean_finalize,
            'std': NoSpecCor_logmel_feature_std_finalize
            }   
    
    print('Calculated zero mean and unit variance Normalization for mel ')
    NoSpecCor_mel_features_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins),'float32')
    for i, filename in enumerate(filename_list_train):
        NoSpecCor_mel_feature_filename = NoSpecCor_mel_feature_path + filename + '.cpickle'
        # load feature
        NoSpecCor_mel_features = pickle.load(open(NoSpecCor_mel_feature_filename, 'rb'))  
        NoSpecCor_mel_features_train[i,:,:] = NoSpecCor_mel_features[:,:]
        
        
    NoSpecCor_mel_feature_mean_finalize = np.mean(NoSpecCor_mel_features_train, axis=(0,2)).reshape(-1,1)
    # Finalize features_std_accummulate
    NoSpecCor_mel_feature_std_finalize = np.std(NoSpecCor_mel_features_train, axis=(0,2)).reshape(-1,1)
    
    NoSpecCor_mel_stats = {'mean': NoSpecCor_mel_feature_mean_finalize,
             'std': NoSpecCor_mel_feature_std_finalize
            }   
     
    ##save feature file
    pickle.dump(NoSpecCor_logmel_stats, open(NoSpecCor_logmel_normalization_filename_path, 'wb' ) )
    pickle.dump(NoSpecCor_mel_stats, open(NoSpecCor_mel_normalization_filename_path, 'wb' ) )
    
    print('NoSpecCor__feature_logmel ----------------')
    print('min = {} '.format(np.min(NoSpecCor_logmel_features_train)))
    print('max = {} '.format(np.max(NoSpecCor_logmel_features_train)))
    print() 
    
    print('NoSpecCor_feature_mean_finalize ')
    print('min = {} '.format(np.min(NoSpecCor_logmel_feature_mean_finalize)))
    print('max = {} '.format(np.max(NoSpecCor_logmel_feature_mean_finalize)))
    print()
    
    print('NoSpecCor__feature_std_finalize')
    print('min = {} '.format(np.min(NoSpecCor_logmel_feature_std_finalize)))
    print('max = {} '.format(np.max(NoSpecCor_logmel_feature_std_finalize)))
    print()
    
    print('NoSpecCor_feature_mel------------------ ')
    print('min = {} '.format(np.min(NoSpecCor_mel_features_train)))
    print('max = {} '.format(np.max(NoSpecCor_mel_features_train)))
    print() 
    
    print('NoSpecCor_feature_mean_finalize ')
    print('min = {} '.format(np.min(NoSpecCor_mel_feature_mean_finalize)))
    print('max = {} '.format(np.max(NoSpecCor_mel_feature_mean_finalize)))
    print()
    
    print('refABC_feature_std_finalize ')
    print('min = {} '.format(np.min(refABC_mel_feature_std_finalize)))
    print('max = {} '.format(np.max(refABC_mel_feature_std_finalize)))
    print() 
    
print('-----DONE') 


## In[] Display spectrogram
#   
#
#displayed_filename_list = [
##                           'airport-helsinki-204-6138-a.cpickle', 
##                          'airport-helsinki-204-6142-a.cpickle', 
##                          'airport-helsinki-204-6143-a.cpickle', 
#                          'shopping_mall-helsinki-129-3849-a.cpickle',
##                          'airport-helsinki-204-6138-b.cpickle', 
##                          'airport-helsinki-204-6142-b.cpickle', 
##                          'airport-helsinki-204-6143-b.cpickle', 
#                          'shopping_mall-helsinki-129-3849-b.cpickle',
##                          'airport-helsinki-204-6138-c.cpickle',
##                          'airport-helsinki-204-6142-c.cpickle', 
##                          'airport-helsinki-204-6143-c.cpickle'
#                          'shopping_mall-helsinki-129-3849-c.cpickle',
#                          ]
#
#refBC_SC_coefficients = pickle.load(open(refBC_spectrumcorrection_filename_path, 'rb'))
#
#refBC_mel_stats = pickle.load(open(refBC_mel_normalization_filename_path, 'rb')) 
#
#refBC_logmel_stats = pickle.load(open(refBC_logmel_normalization_filename_path, 'rb')) 
##
#refABC_SC_coefficients = pickle.load(open(refABC_spectrumcorrection_filename_path, 'rb'))
#
#refABC_mel_stats = pickle.load(open(refABC_mel_normalization_filename_path, 'rb'))
#
#refABC_logmel_stats = pickle.load(open(refABC_logmel_normalization_filename_path, 'rb'))
##
#NoSpecCor_mel_stats = pickle.load(open(NoSpecCor_mel_normalization_filename_path, 'rb'))
#
#NoSpecCor_logmel_stats = pickle.load(open(NoSpecCor_logmel_normalization_filename_path, 'rb'))
#
#
#
#print('Spectrum correction using ref of BC')
#print('=====================')
#tmp_refBC_normed_man_logmel = np.zeros((3, 256, 862),'float32')
#tmp_refBC_normed_logmel = np.zeros((3, 256, 862),'float32')
#for i, filename in enumerate(displayed_filename_list):
#
#    spectrum_filename_path = spectrum_path + filename
#    refBC_logmel_feature_filename_path = refBC_logmel_feature_path + filename
#    refBC_mel_feature_filename_path = refBC_mel_feature_path + filename
#    
#    spectrum = pickle.load(open(spectrum_filename_path, 'rb'))   
#    refBC_corrected_spectrum = spectrum * refBC_SC_coefficients
#    
#    refBC_mel_feature = pickle.load(open(refBC_mel_feature_filename_path, 'rb'))
#    refBC_norm_man_logmel_feature = np.log(np.abs((refBC_mel_feature - refBC_mel_stats['mean'])/(refBC_mel_stats['std'] + eps))+eps)
#    tmp_refBC_normed_man_logmel[i,:,:] = refBC_norm_man_logmel_feature
#    
#    refBC_logmel_feature = pickle.load(open(refBC_logmel_feature_filename_path, 'rb'))
#    refBC_norm_logmel_feature =(refBC_logmel_feature - refBC_logmel_stats['mean'])/(refBC_logmel_stats['std']+ eps)
#    tmp_refBC_normed_logmel[i,:,:] = refBC_norm_logmel_feature
#    
#    if (refBC_norm_man_logmel_feature - refBC_norm_logmel_feature).all()<0.000001:
#        print ('Same for log mel and manual logmel')
#    else:
#        print ('Different for log mel and manual logmel')
#
## Creates four polar axes, and accesses them through the returned array
#    plt.figure(i+1,figsize=(12, 3))
#
#    plt.subplot(1,4,1)
#    librosa.display.specshow(np.log(spectrum), x_axis='time')
#    plt.title('Spectrum')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
##    
#    plt.subplot(1,4,2)
#    librosa.display.specshow(np.log(refBC_corrected_spectrum), x_axis='time')
#    plt.title('corrected_spectrum')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
#    plt.subplot(1,4,3)
#    librosa.display.specshow(refBC_norm_man_logmel_feature, x_axis='time')
#    plt.title('norm_man_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
#    plt.subplot(1,4,4)
#    librosa.display.specshow(refBC_norm_logmel_feature, x_axis='time')
#    plt.title('norm_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#
##if (tmp_refBC_normed_logmel[0,:,:] - tmp_refBC_normed_logmel[1,:,:]).all() > 1:
##    print('tmp_refBC_normed_logmel[0,:,:] - tmp_refBC_normed_logmel[1,:,:]).all() > 1')
##else:
##    print('tmp_refBC_normed_logmel[0,:,:] - tmp_refBC_normed_logmel[1,:,:]).all() < 1')
##
##if (tmp_refBC_normed_man_logmel[0,:,:] - tmp_refBC_normed_man_logmel[1,:,:]).all() > 1:
##    print('tmp_refBC_normed_man_logmel[0,:,:] - tmp_refBC_normed_man_logmel[1,:,:]).all() > 1')
##else:
##    print('tmp_refBC_normed_man_logmel[0,:,:] - tmp_refBC_normed_man_logmel[1,:,:]).all() < 1')
#    
#  
#print('tmp_refBC_normed_man_logmel')
#print('tmp_refBC_normed_man_logmel[0,:,:] - tmp_refBC_normed_man_logmel[1,:,:]')
#print(tmp_refBC_normed_man_logmel[0,:,:] - tmp_refBC_normed_man_logmel[1,:,:])
#print()
#print('tmp_refBC_normed_man_logmel[0,:,:] - tmp_refBC_normed_man_logmel[2,:,:]') 
#print(tmp_refBC_normed_man_logmel[0,:,:] - tmp_refBC_normed_man_logmel[2,:,:])  
#print()
#print('tmp_refBC_normed_logmel')
#print('tmp_refBC_normed_logmel[0,:,:] - tmp_refBC_normed_logmel[1,:,:]')
#print(tmp_refBC_normed_logmel[0,:,:] - tmp_refBC_normed_logmel[1,:,:])
#print()
#print('tmp_refBC_normed_logmel[0,:,:] - tmp_refBC_normed_logmel[2,:,:]')  
#print(tmp_refBC_normed_logmel[0,:,:] - tmp_refBC_normed_logmel[2,:,:])
#print()
#
# 
#
#print('Spectrum correction using ref of ABC')
#print('=====================')
#tmp_refABC_normed_man_logmel = np.zeros((3, 256, 862),'float32')
#tmp_refABC_normed_logmel = np.zeros((3, 256, 862),'float32')
#for i, filename in enumerate(displayed_filename_list):
#
#    spectrum_filename_path = spectrum_path + filename
#    refABC_logmel_feature_filename_path = refABC_logmel_feature_path + filename
#    refABC_mel_feature_filename_path = refABC_mel_feature_path + filename
#    
#    spectrum = pickle.load(open(spectrum_filename_path, 'rb'))   
#    refABC_corrected_spectrum = spectrum * refABC_SC_coefficients
#    
#    refABC_mel_feature = pickle.load(open(refABC_mel_feature_filename_path, 'rb'))
#    refABC_norm_man_logmel_feature = np.log(np.abs((refABC_mel_feature - refABC_mel_stats['mean'])/(refABC_mel_stats['std']+eps))+eps)
#    tmp_refABC_normed_man_logmel[i,:,:] = refABC_norm_man_logmel_feature
#    
#    refABC_logmel_feature = pickle.load(open(refABC_logmel_feature_filename_path, 'rb'))
#    refABC_norm_logmel_feature =(refABC_logmel_feature - refABC_logmel_stats['mean'])/(refABC_logmel_stats['std']+eps)
#    tmp_refABC_normed_logmel[i,:,:] = refABC_norm_logmel_feature
#    
#    if (refABC_norm_man_logmel_feature - refABC_norm_logmel_feature).all()<0.000001:
#        print ('Same for log mel and manual logmel')
#    else:
#        print ('Different for log mel and manual logmel')
#
## Creates four polar axes, and accesses them through the returned array
#    plt.figure(i+4,figsize=(12, 3))
#
#    plt.subplot(1,4,1)
#    librosa.display.specshow(np.log(spectrum), x_axis='time')
#    plt.title('Spectrum')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
##    
#    plt.subplot(1,4,2)
#    librosa.display.specshow(np.log(refABC_corrected_spectrum), x_axis='time')
#    plt.title('corrected_spectrum')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#
#    
#    plt.subplot(1,4,3)
#    librosa.display.specshow(refABC_norm_man_logmel_feature, x_axis='time')
#    plt.title('norm_man_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
#    plt.subplot(1,4,4)
#    librosa.display.specshow(refABC_norm_logmel_feature, x_axis='time')
#    plt.title('norm_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
##if (tmp_refABC_normed_logmel[0,:,:] - tmp_refABC_normed_logmel[1,:,:]).all() > 1:
##    print('tmp_refABC_normed_logmel[0,:,:] - tmp_refABC_normed_logmel[1,:,:]).all() > 1')
##else:
##    print('tmp_refABC_normed_logmel[0,:,:] - tmp_refABC_normed_logmel[1,:,:]).all() < 1')
##
##if (tmp_refABC_normed_man_logmel[0,:,:] - tmp_refABC_normed_man_logmel[1,:,:]).all() > 1:
##    print('tmp_refABC_normed_man_logmel[0,:,:] - tmp_refABC_normed_man_logmel[1,:,:]).all() > 1')
##else:
##    print('tmp_refABC_normed_man_logmel[0,:,:] - tmp_refABC_normed_man_logmel[1,:,:]).all() < 1')
#
#print('tmp_refABC_normed_man_logmel')
#print('tmp_refABC_normed_man_logmel[0,:,:] - tmp_refABC_normed_man_logmel[1,:,:]')
#print(tmp_refABC_normed_man_logmel[0,:,:] - tmp_refABC_normed_man_logmel[1,:,:])
#print()
#print('tmp_refABC_normed_man_logmel[0,:,:] - tmp_refABC_normed_man_logmel[2,:,:]')  
#print(tmp_refABC_normed_man_logmel[0,:,:] - tmp_refABC_normed_man_logmel[2,:,:]) 
#print()
#print('tmp_refABC_normed_logmel')
#print('tmp_refABC_normed_logmel[0,:,:] - tmp_refABC_normed_logmel[1,:,:]')
#print(tmp_refABC_normed_logmel[0,:,:] - tmp_refABC_normed_logmel[1,:,:])
#print()
#print('tmp_refABC_normed_logmel[0,:,:] - tmp_refABC_normed_logmel[2,:,:]')  
#print(tmp_refABC_normed_logmel[0,:,:] - tmp_refABC_normed_logmel[2,:,:]) 
#print()
#
#
#print('No spectrum correction')
#print('====================='  )  
#tmp_spectrum = np.zeros((3, 1025, 862),'float32')
#tmp_NoSpecCor_mel_feature = np.zeros((3, 256, 862),'float32')
#tmp_NoSpecCor_normed_man_logmel = np.zeros((3, 256, 862),'float32')
#tmp_NoSpecCor_normed_logmel = np.zeros((3, 256, 862),'float32')
#ext_tmp_NoSpecCor_normed_man_logmel = np.zeros((3, 256, 862),'float32')
#ext_tmp_NoSpecCor_normed_logmel = np.zeros((3, 256, 862),'float32')
#
#for i, filename in enumerate(displayed_filename_list):
#
#    spectrum_filename_path = spectrum_path + filename
#    NoSpecCor_logmel_feature_filename_path = NoSpecCor_logmel_feature_path + filename
#    NoSpecCor_mel_feature_filename_path = NoSpecCor_mel_feature_path + filename
#    
#    spectrum = pickle.load(open(spectrum_filename_path, 'rb'))   
#    tmp_spectrum[i,:,:] =spectrum
#    #
#    NoSpecCor_mel_feature = pickle.load(open(NoSpecCor_mel_feature_filename_path, 'rb'))
#    tmp_NoSpecCor_mel_feature[i,:,:] =NoSpecCor_mel_feature
#
#    NoSpecCor_norm_man_logmel_feature = np.log(np.abs((NoSpecCor_mel_feature - NoSpecCor_mel_stats['mean'])/(NoSpecCor_mel_stats['std'] + eps))+ eps)
#    tmp_NoSpecCor_normed_man_logmel[i,:,:] = NoSpecCor_norm_man_logmel_feature
#    #
#    NoSpecCor_logmel_feature = pickle.load(open(NoSpecCor_logmel_feature_filename_path, 'rb'))
#    NoSpecCor_norm_logmel_feature =(NoSpecCor_logmel_feature - NoSpecCor_logmel_stats['mean'])/(NoSpecCor_logmel_stats['std']+ eps)
#    tmp_NoSpecCor_normed_logmel[i,:,:] = NoSpecCor_norm_logmel_feature
#    
#    if (refBC_norm_man_logmel_feature - refBC_norm_logmel_feature).all()<0.000001:
#        print ('Same for log mel and manual logmel')
#    else:
#        print ('Different for log mel and manual logmel')
#
## Creates four polar axes, and accesses them through the returned array
#    plt.figure(i+7,figsize=(12, 3))
#
#    plt.subplot(1,5,1)
#    librosa.display.specshow(np.log(spectrum), x_axis='time')
#    plt.title('Spectrum')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#
#    plt.subplot(1,5,2)
#    librosa.display.specshow(NoSpecCor_norm_man_logmel_feature, x_axis='time')
#    plt.title('norm_man_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
#    plt.subplot(1,5,3)
#    librosa.display.specshow(NoSpecCor_norm_logmel_feature, x_axis='time')
#    plt.title('norm_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
#    ####
#    mel_basis = librosa.filters.mel(sr    = sr,
#                                       n_fft  = NumFFTPoints,
#                                       n_mels = NumFreqBins,
#                                       fmin   = 0.0,
#                                       fmax   = sr/2,
#                                       htk    = True,
#                                       norm   = None)
#                                     
#    ext_NoSpecCor_mel_feature = np.dot(mel_basis, spectrum)
#    ext_NoSpecCor_norm_man_logmel_feature = np.log(np.abs((ext_NoSpecCor_mel_feature - NoSpecCor_mel_stats['mean'])/(NoSpecCor_mel_stats['std'] + eps))+ eps)
#    ext_tmp_NoSpecCor_normed_man_logmel[i,:,:] = ext_NoSpecCor_norm_man_logmel_feature
#    
#    ext_NoSpecCor_logmel_feature = np.log(np.abs(ext_NoSpecCor_mel_feature) + eps)
#    ext_NoSpecCor_norm_logmel_feature =(ext_NoSpecCor_logmel_feature - NoSpecCor_logmel_stats['mean'])/(NoSpecCor_logmel_stats['std']+ eps)
#    ext_tmp_NoSpecCor_normed_logmel[i,:,:] = ext_NoSpecCor_norm_logmel_feature
#    
#    plt.subplot(1,5,4)
#    librosa.display.specshow(ext_NoSpecCor_norm_man_logmel_feature, x_axis='time')
#    plt.title('ext_norm_man_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
#    plt.subplot(1,5,5)
#    librosa.display.specshow(ext_NoSpecCor_norm_logmel_feature, x_axis='time')
#    plt.title('ext_norm_logmel_feature')
#    plt.colorbar(format='%+2.0f dB')
##    plt.tight_layout()
#    
#print()    
#print('tmp_spectrum')    
#print('tmp_spectrum[0,:,:] -tmp_spectrum[1,:,:]')   
#print(tmp_spectrum[0,:,:] -tmp_spectrum[1,:,:])   
#print()
#print('tmp_spectrum[0,:,:] -tmp_spectrum[2,:,:]')   
#print(tmp_spectrum[0,:,:] -tmp_spectrum[2,:,:])   
#print() 
#
#print('tmp_NoSpecCor_mel_feature')
#print('tmp_NoSpecCor_mel_feature[0,:,:] - tmp_NoSpecCor_mel_feature[1,:,:]')
#print(tmp_NoSpecCor_mel_feature[0,:,:] - tmp_NoSpecCor_mel_feature[1,:,:])
#print()
#print('tmp_NoSpecCor_mel_feature[0,:,:] - tmp_NoSpecCor_mel_feature[2,:,:]')
#print(tmp_NoSpecCor_mel_feature[0,:,:] - tmp_NoSpecCor_mel_feature[2,:,:])
#print()  
#
#print('tmp_NospecCor_normed_man_logmel')
#print('tmp_NospecCor_normed_man_logmel[0,:,:] - tmp_NospecCor_normed_man_logmel[1,:,:]')
#print(tmp_NoSpecCor_normed_man_logmel[0,:,:] - tmp_NoSpecCor_normed_man_logmel[1,:,:])
#print()
#print('tmp_NospecCor_normed_man_logmel[0,:,:] - tmp_NospecCor_normed_man_logmel[2,:,:]')
#print(tmp_NoSpecCor_normed_man_logmel[0,:,:] - tmp_NoSpecCor_normed_man_logmel[2,:,:])  
#print()
#
#print('tmp_NospecCor_normed_logmel')
#print('tmp_NospecCor_normed_logmel[0,:,:] - tmp_NospecCor_normed_logmel[1,:,:]')
#print(tmp_NoSpecCor_normed_logmel[0,:,:] - tmp_NoSpecCor_normed_logmel[1,:,:])
#print()
#print('tmp_NospecCor_normed_logmel[0,:,:] - tmp_NospecCor_normed_logmel[2,:,:]') 
#print(tmp_NoSpecCor_normed_logmel[0,:,:] - tmp_NoSpecCor_normed_logmel[2,:,:])
#print()
#
#
#print('ext_tmp_NospecCor_normed_man_logmel')
#print('ext_tmp_NospecCor_normed_man_logmel[0,:,:] - ext_tmp_NospecCor_normed_man_logmel[1,:,:]')
#print(ext_tmp_NoSpecCor_normed_man_logmel[0,:,:] - ext_tmp_NoSpecCor_normed_man_logmel[1,:,:])
#print()
#print('ext_tmp_NospecCor_normed_man_logmel[0,:,:] - ext_tmp_NospecCor_normed_man_logmel[2,:,:]')
#print(ext_tmp_NoSpecCor_normed_man_logmel[0,:,:] - ext_tmp_NoSpecCor_normed_man_logmel[2,:,:])  
#print()
#
#print('ext_tmp_NospecCor_normed_logmel')
#print('ext_tmp_NospecCor_normed_logmel[0,:,:] - ext_tmp_NospecCor_normed_logmel[1,:,:]')
#print(ext_tmp_NoSpecCor_normed_logmel[0,:,:] - ext_tmp_NoSpecCor_normed_logmel[1,:,:])
#print()
#print('ext_tmp_NospecCor_normed_logmel[0,:,:] - ext_tmp_NospecCor_normed_logmel[2,:,:]') 
#print(ext_tmp_NoSpecCor_normed_logmel[0,:,:] - ext_tmp_NoSpecCor_normed_logmel[2,:,:])
#print()
#
#plt.show()


# In[ ]: Prepare data for training 


print('Training')
print('========================================')
print('-----Loading data for training')

normalization_filename_path_list = [
        refBC_mel_normalization_filename_path,
        refBC_logmel_normalization_filename_path,
#        refABC_mel_normalization_filename_path,
#        refABC_logmel_normalization_filename_path,
        NoSpecCor_mel_normalization_filename_path,
        NoSpecCor_logmel_normalization_filename_path,
        
        ]
feature_path_list = [
        refBC_mel_feature_path,
        refBC_logmel_feature_path,
#        refABC_mel_feature_path,
#        refABC_logmel_feature_path,
        NoSpecCor_mel_feature_path,
        NoSpecCor_logmel_feature_path,
        ]

for model_name_, normalization_filename_path,feature_path in zip(
        model_names, normalization_filename_path_list, feature_path_list):
    stats = pickle.load(open(normalization_filename_path, 'rb')) 
    #load log-mel spectrograms
    LM_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
    if '_logmel_' in model_name_:
        for i, filename in enumerate(filename_list_train):    
            feature_filename = feature_path + filename + '.cpickle'
            feature = pickle.load(open(feature_filename, 'rb'))
            if normalization_flag == True:
                feature = (feature -  stats['mean'])/(stats['std']+eps)
            LM_train[i,:,:,:]= feature[:,:,None]
        
        LM_val = np.zeros((len(wavpaths_val),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
        for i, filename in enumerate(filename_list_val):    
            feature_filename = feature_path + filename + '.cpickle'
            feature = pickle.load(open(feature_filename, 'rb'))
            if normalization_flag == True:
                feature = (feature -  stats['mean'])/(stats['std']+eps)
            LM_val[i,:,:,:]= feature[:,:,None]
            
    elif '_mel_' in model_name_:
        for i, filename in enumerate(filename_list_train):    
            feature_filename = feature_path + filename + '.cpickle'
            feature = pickle.load(open(feature_filename, 'rb'))
            if normalization_flag == True:
                feature = (feature -  stats['mean'])/(stats['std']+eps)
            #
            feature = np.log(np.abs(feature) + eps)
            
            LM_train[i,:,:,:]= feature[:,:,None]
        
        LM_val = np.zeros((len(wavpaths_val),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
        for i, filename in enumerate(filename_list_val):    
            feature_filename = feature_path + filename + '.cpickle'
            feature = pickle.load(open(feature_filename, 'rb'))
            if normalization_flag == True:
                feature = (feature -  stats['mean'])/(stats['std']+eps)
            #
            feature = np.log(np.abs(feature) + eps)
            LM_val[i,:,:,:]= feature[:,:,None]
    
    print('-----DONE')
    print('Training set information')
    print('Training set size {}'.format(LM_train.shape))
    print()
    print('Validation(Test) set information')
    print('Validation set size {}'.format(LM_val.shape))
    print()
    print('Min Training set = {}'.format(np.min(LM_train)))
    print('Max Training set = {}'.format(np.max(LM_train)))
    print()
    print('Min Test set = {}'.format(np.min(LM_val)))
    print('Max Test set = {}'.format(np.max(LM_val)))
       
    # In[ ]: Compile model
     
           
    model_structures = [
#            model_best2019_base1(LM_train,n_labels=NumClasses, wd=1e-3),
#            model_best2019_base2(LM_train,n_labels=NumClasses, wd=1e-3)
            
            create_model_db_3cnn_nexp(LM_train, BNaxis=1, n_layerexperts=10, n_labels=NumClasses, name='db_3cnn')                          
            ]
    for model_structure_  in model_structures:
        
        model_filename_path = model_path + model_name_   #'model_best2019_base2.h5'
    
        if not os.path.isfile(model_filename_path):        
            #create and compile the model
            model = model_structure_
    #        model.compile(loss='categorical_crossentropy',
    #                      optimizer = Adadelta(lr=max_lr),
    #                      metrics=['accuracy'])
            model.compile(loss= categorical_focal_loss(gamma=1., alpha=1),
                          optimizer = Adadelta(lr=max_lr),
                          metrics=['accuracy'])
            model.summary()
            
            
            # In[ ]: Train model
            
            
            #set learning rate schedule
            #lr_scheduler = LR_WarmRestart(nbatch=np.ceil(LM_train.shape[0]/batch_size), Tmult=2,
            #                              initial_lr=max_lr, min_lr=max_lr*1e-4,
            #                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0]) 
            checkpoint = ModelCheckpoint(model_filename_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            ##ProgressLoggerCallback, StasherCallback=checkpoint
            callbacks = [checkpoint]
            
            #create data generator
            TrainDataGen = MixupGenerator(LM_train, 
                                          y_train, 
                                          batch_size=batch_size,
                                          alpha=mixup_alpha,
                                          )()
            
            #train the model
            history = model.fit_generator(TrainDataGen,
                                          validation_data=(LM_val, y_val),
                                          epochs=num_epochs, 
                                          verbose=2, 
                                          callbacks=callbacks,
                                          steps_per_epoch=np.ceil(LM_train.shape[0]/batch_size)
                                          ) 
            
            
            # In[ ]: Save model
            
            
            model.save(model_filename_path)
        print('-----DONE') 
    
    
    # In[ ]: Inferrence
    
    
    #load filenames and labels
    dev_test_df = pd.read_csv(ValFile,sep='\t', encoding='ASCII')
    Inds_device_a=np.where(dev_test_df['filename'].str.contains("-a.wav")==True)[0]
    Inds_device_b=np.where(dev_test_df['filename'].str.contains("-b.wav")==True)[0]
    Inds_device_c=np.where(dev_test_df['filename'].str.contains("-c.wav")==True)[0]
    Inds_device_bc=np.concatenate((Inds_device_b,Inds_device_c),axis=-1)
    
    wavpaths = dev_test_df['filename'].tolist()
    ClassNames = np.unique(dev_test_df['scene_label'])
    y_val_labels =  dev_test_df['scene_label'].astype('category').cat.codes.values
    
    
    # In[5]:
    
    
##    for model_name_ in model_names: 
#             
#        # load model
#        # Get model filename
#        model_filename_path = model_path + model_name_ 
#        # Initialize model to None, load when first non-tested file encountered.
#        keras_model = None
#        with CustomObjectScope({'DenseMoE': DenseMoE},
#                               {'categorical_focal_loss_fixed': categorical_focal_loss(gamma=1., alpha=1.)},
#                                           ):
#            keras_model = keras.models.load_model(model_filename_path)
        
#         #Get results filename
#        #result_filename_path = result_path + os.path.splitext()  #'model_best2019_base1.cpickle'
        
        
    # In[6]:
    
 
    #load and run the model
    with CustomObjectScope({'DenseMoE': DenseMoE},
                               {'categorical_focal_loss_fixed': categorical_focal_loss(gamma=1., alpha=1.)},
                                           ):
        best_model = keras.models.load_model(model_filename_path)
    y_pred_val = np.argmax(best_model.predict(LM_val),axis=1)
    
    
    # In[7]:
    
    print()
    print('Model name:{}'.format(model_name_))
    #get metrics for all devices combined
    Overall_accuracy = np.sum(y_pred_val==y_val_labels)/LM_val.shape[0]
    print('All devices')
    print("overall accuracy: ", Overall_accuracy)
    print()
#    plot_confusion_matrix(y_val_labels, y_pred_val, ClassNames,normalize=True,title="Task 1b, all devices")
#    
#    conf_matrix = confusion_matrix(y_val_labels,y_pred_val)
#    conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
#    conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
#    recall_by_class = np.diagonal(conf_mat_norm_recall)
#    precision_by_class = np.diagonal(conf_mat_norm_precision)
#    mean_recall = np.mean(recall_by_class)
#    mean_precision = np.mean(precision_by_class)
#    
#    print("per-class accuracy (recall): ",recall_by_class)
#    print("per-class precision: ",precision_by_class)
#    print("mean per-class recall: ",mean_recall)
#    print("mean per-class precision: ",mean_precision)
#    
    
    # In[8]:
    
    
    #get metrics for device A only
    Overall_accuracy = np.sum(y_pred_val[Inds_device_a]==y_val_labels[Inds_device_a])/len(Inds_device_a)
    print('Device A ')
    print("overall accuracy: ", Overall_accuracy)
    print()
#    plot_confusion_matrix(y_val_labels[Inds_device_a], y_pred_val[Inds_device_a], ClassNames,normalize=True,title="Task 1b, Device A")
#    
#    conf_matrix = confusion_matrix(y_val_labels[Inds_device_a],y_pred_val[Inds_device_a])
#    conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
#    conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
#    recall_by_class = np.diagonal(conf_mat_norm_recall)
#    precision_by_class = np.diagonal(conf_mat_norm_precision)
#    mean_recall = np.mean(recall_by_class)
#    mean_precision = np.mean(precision_by_class)
#    
#    print("per-class accuracy (recall): ",recall_by_class)
#    print("per-class precision: ",precision_by_class)
#    print("mean per-class recall: ",mean_recall)
#    print("mean per-class precision: ",mean_precision)
#    
    
    # In[9]:
    
    
    #get metrics for device B only
    Overall_accuracy = np.sum(y_pred_val[Inds_device_b]==y_val_labels[Inds_device_b])/len(Inds_device_b)
    print('Device B ')
    print("overall accuracy: ", Overall_accuracy)
    print()
#    plot_confusion_matrix(y_val_labels[Inds_device_b], y_pred_val[Inds_device_b], ClassNames,normalize=True,title="Task 1b, Device B")
#    
#    conf_matrix = confusion_matrix(y_val_labels[Inds_device_b],y_pred_val[Inds_device_b])
#    conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
#    conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
#    recall_by_class = np.diagonal(conf_mat_norm_recall)
#    precision_by_class = np.diagonal(conf_mat_norm_precision)
#    mean_recall = np.mean(recall_by_class)
#    mean_precision = np.mean(precision_by_class)
#    
#    print("per-class accuracy (recall): ",recall_by_class)
#    print("per-class precision: ",precision_by_class)
#    print("mean per-class recall: ",mean_recall)
#    print("mean per-class precision: ",mean_precision)
    
    
    # In[10]:
    
    
    #get metrics for device C only
    Overall_accuracy = np.sum(y_pred_val[Inds_device_c]==y_val_labels[Inds_device_c])/len(Inds_device_c)
    print('Device C')
    print("overall accuracy: ", Overall_accuracy)
    print()
#    plot_confusion_matrix(y_val_labels[Inds_device_c], y_pred_val[Inds_device_c], ClassNames,normalize=True,title="Task 1b, Device C")
#    
#    conf_matrix = confusion_matrix(y_val_labels[Inds_device_c],y_pred_val[Inds_device_c])
#    conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
#    conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
#    recall_by_class = np.diagonal(conf_mat_norm_recall)
#    precision_by_class = np.diagonal(conf_mat_norm_precision)
#    mean_recall = np.mean(recall_by_class)
#    mean_precision = np.mean(precision_by_class)
#    
#    print("per-class accuracy (recall): ",recall_by_class)
#    print("per-class precision: ",precision_by_class)
#    print("mean per-class recall: ",mean_recall)
#    print("mean per-class precision: ",mean_precision)
#    
    
    # In[11]:
    
    
    #get metrics for device B and C 
    Overall_accuracy = np.sum(y_pred_val[Inds_device_bc]==y_val_labels[Inds_device_bc])/len(Inds_device_bc)
    print('Device B and C')
    print("overall accuracy: ", Overall_accuracy)
    print()
#    plot_confusion_matrix(y_val_labels[Inds_device_bc], y_pred_val[Inds_device_bc], ClassNames,normalize=True,title="Task 1b, Device B and C")
#    
#    conf_matrix = confusion_matrix(y_val_labels[Inds_device_bc],y_pred_val[Inds_device_bc])
#    conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
#    conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
#    recall_by_class = np.diagonal(conf_mat_norm_recall)
#    precision_by_class = np.diagonal(conf_mat_norm_precision)
#    mean_recall = np.mean(recall_by_class)
#    mean_precision = np.mean(precision_by_class)
#    
#    print("per-class accuracy (recall): ",recall_by_class)
#    print("per-class precision: ",precision_by_class)
#    print("mean per-class recall: ",mean_recall)
#    print("mean per-class precision: ",mean_precision)
#    
    
    print('Done')