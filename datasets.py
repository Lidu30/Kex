import torch
from torchvision import datasets, transforms
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader, TensorDataset
import mne
from mne.preprocessing import ICA
from scipy.signal import istft
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import hickle

def EEG_generator (batch_size):
    train_X_train = np.load ("/mnt/data7_4T/temp/yikai/RPA_AUC_segment/tuh_stft_dev12_clean_merge/totalx.npy")[10000:]
    train_y_train = np.load ("/mnt/data7_4T/temp/yikai/RPA_AUC_segment/tuh_stft_dev12_clean_merge/totaly.npy")[10000:]

    ones_indices = np.where(train_y_train == 1)[0] #indices.

    final_data = train_X_train[ones_indices]
    final_data_Y = train_y_train [ones_indices]

    duplicated_ones_indices = np.repeat(final_data_Y,4)
    duplicated_train_X_train = np.repeat(final_data,4,axis= 0)

    train_X_train = np.concatenate ((train_X_train,duplicated_train_X_train),axis=0)
    train_y_train = np.concatenate ((train_y_train,duplicated_ones_indices),axis=0)

    train_y_train = train_y_train.astype(np.int64)

    print (train_X_train.shape)
    print (train_y_train.shape)

    print("Number of 1s:", np.count_nonzero(train_y_train == 1))
    print("Number of 0s:", np.count_nonzero(train_y_train == 0))

    test_X_train = np.load("/mnt/data7_4T/temp/yikai/RPA_AUC_segment/tuh_stft_dev12_clean_merge/totalx.npy")[:10000]
    test_y_train = np.load("/mnt/data7_4T/temp/yikai/RPA_AUC_segment/tuh_stft_dev12_clean_merge/totaly.npy")[:10000]

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_y_train = test_y_train.astype(np.int64)

    train_X_train = np.transpose(train_X_train, (0, 3, 2, 1))
    test_X_train = np.transpose(test_X_train, (0, 3, 2, 1))


    train_X_train = train_X_train.reshape(train_X_train.shape[0],-1)
    print(train_X_train.shape)

    test_X_train = test_X_train.reshape(test_X_train.shape[0],-1)

    train_dataset = TensorDataset(torch.FloatTensor(train_X_train), torch.tensor(train_y_train))
    test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

    train_loader = DataLoader(train_dataset,batch_size=batch_size,
    shuffle=True)

    test_loader = DataLoader(test_dataset,batch_size=batch_size,
    shuffle=False)

    n_classes = 2
    seq_length = 23*125
    input_channels = 19

    return train_loader, test_loader, seq_length, input_channels, n_classes

def EEG_generator_Time (batch_size):

        ICA = True

        file_name_x = "/mnt/data12_16T/TUH_ICA_No_STFT/train/"
        dev_path = "/mnt/data12_16T/TUH_ICA_No_STFT/dev/"

        train_y_path = "/mnt/data12_16T/thomas/TUH_preprocessed/train_y.npy"
        # train_path_x = "/mnt/data12_16T/thomas/TUH_preprocessed/train_x.npy"  # IF WILLING to do a lot

        dev_path_x = "/mnt/data12_16T/thomas/TUH_preprocessed/dev_x.npy" #NO ICA
        dev_path_y = "/mnt/data12_16T/thomas/TUH_preprocessed/dev_y.npy"

        if ICA:
            train_X_train = ICA_Data(file_name_x)
            test_X_train = ICA_Data(dev_path)
            test_X_train = test_X_train.astype(np.float16)

        train_X_train = train_X_train[0:70000]  # Good
        train_y_train = np.load(train_y_path)[0:70000]  # Good

        train_y_train = train_y_train.astype(np.int64)

        print(train_X_train.shape, train_y_train.shape)

        print("Number of 1s:", np.count_nonzero(train_y_train == 1))
        print("Number of 0s:", np.count_nonzero(train_y_train == 0))

        def Noise_Min_Max(X_test):

            new_or = X_test.reshape(-1, X_test.shape[-1])
            scaler = MinMaxScaler()
            new_or = scaler.fit_transform(new_or)

            sfreq = 250  # Sample frequency in Hz, adjust as needed
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4',
                   u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ',
                   u'P7-T7']

            info1 = mne.create_info(chs, sfreq, ch_types='eeg')
            raw1 = mne.io.RawArray(new_or.transpose(1, 0), info1)
            raw1.notch_filter(60, fir_design='firwin')

            filtered_data = raw1.get_data().transpose(1, 0)
            filtered_data = filtered_data.reshape(X_test.shape[0], 3000, X_test.shape[2])

            return filtered_data

        train_X_train = Noise_Min_Max(train_X_train)
        train_X_train = train_X_train.astype(np.float16)
        test_X_train = Noise_Min_Max(test_X_train[0:20000])
        test_X_train = test_X_train.astype(np.float16)

        # test_X_train = test_X_train[0:20000] #good
        test_y_train = np.load(dev_path_y)[0:20000] #good
        test_y_train = test_y_train.astype(np.int64)

        print("Number of 1s:", np.count_nonzero(test_y_train == 1))
        print("Number of 0s:", np.count_nonzero(test_y_train == 0))

        train_dataset = TensorDataset(torch.FloatTensor(train_X_train), torch.tensor(train_y_train))
        test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # I m gonna use 120.
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # i m gonna use 120

        n_classes = 2
        seq_length = 12 * 250
        input_channels = 19

        return train_loader, test_loader, seq_length, input_channels, n_classes

def create_mne_raw(data, sfreq, chs=None):

    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1',
           'O2']

    if chs is None:
        chs_ = ['ch{}'.format(i) for i in range(data.shape[0])]
    else:
        # assert data.shape[0] == len(chs)
        chs_ = ch_names

    ch_types = ['eeg' for _ in range(len(chs_))]

    info = mne.create_info(ch_names=chs_, sfreq=sfreq, ch_types=ch_types, verbose=False)
    print (info)
    raw = mne.io.RawArray(data * 1e-7, info)
    print ("here1")

    return raw

def ica_arti_remove(data, sfreq, chs=None):

    raw = create_mne_raw(data, sfreq, chs)
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=0.1, h_freq=None, verbose=False)

    ica = ICA(n_components=19, random_state=13)
    try:
        ica.fit(filt_raw, verbose=False)
    except:
        return None

    print ("here2")

    ica.exclude = []

    eog_indices1, eog_scores1 = ica.find_bads_eog(filt_raw, threshold=2, ch_name='Fp1', verbose=False)
    print('eog_indices1', eog_indices1)
    eog_indices2, eog_scores2 = ica.find_bads_eog(filt_raw, threshold=2, ch_name='Fp2', verbose=False)
    print('eog_indices2', eog_indices2)

    if len(eog_indices1) > 0:
        ica.exclude.append(eog_indices1[0])
    if len(eog_indices2) > 0:
        ica.exclude.append(eog_indices2[0])

    print('ica.exclude', ica.exclude)

    if len(ica.exclude) > 0:
        reconst_raw = filt_raw.copy()
        reconst_raw.load_data()
        ica.apply(reconst_raw)
        print('Reconstructing data from ICA components...')
        return reconst_raw.get_data() * 1e6

    return data

def Segmentation (data_processed):

    segment_duration_samples = 12 * 250
    num_segments = int (data_processed.shape[1]//segment_duration_samples)
    segmented_data_shape = (19, 3000, num_segments)
    segmented_data = np.zeros(segmented_data_shape)

    for i in range(num_segments):

        start_idx = i * segment_duration_samples
        end_idx = (i + 1) * segment_duration_samples
        segment = data_processed[:, start_idx:end_idx]
        segmented_data[:,:,i] = segment

    segmented_data = segmented_data.transpose(2, 1, 0)

    return segmented_data

def initialize_savings (data, output_fol):

    start_idx1= 1000
    print (data.shape[0])
    n = int(data.shape[0]/start_idx1)
    for i in range(n):
        start_idx = i * start_idx1 #0,
        end_idx = (i + 1) * start_idx1 #1000
        data1 = data[start_idx:end_idx]
        data2 = data1.transpose(2, 0, 1).reshape(19, -1) #19, 3000 x length.
        data_processed = ica_arti_remove(data2, 250, chs=19)
        segmented_data = Segmentation(data_processed)
        segmented_data = segmented_data.astype(np.float16)
        np.save (output_fol + str(i) +"_subset.npy", segmented_data)

def extract_number(filename):
    """Extracts the number from the filename."""
    match = re.search(r'(\d+)_subset', filename)
    return int(match.group(1)) if match else None

def ICA_Data(directory):

    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    files.sort(key=extract_number)

    # List to store the loaded arrays
    arrays = []

    # Loop through each sorted file and load the array
    for filename in files:
        file_path = os.path.join(directory, filename)
        print (filename)
        data = np.load(file_path)
        arrays.append(data)

    # Stack the arrays along the first dimension
    final_array = np.vstack(arrays)

    # Print the final shape
    print("Final shape:", final_array.shape)
    return final_array


def Epilepsia_12s_STFT(patname,batch_size):

    TestX = hickle.load ("/mnt/data7_4T/temp/yikai/EpilepsiaSurfS_12s_19ch_stft/bckg_" + patname + ".hickle")
    TestY = hickle.load ("/mnt/data7_4T/temp/yikai/EpilepsiaSurfS_12s_19ch_stft/seiz_" + patname + ".hickle")

    TestX1 = np.concatenate([TestX[0][i] for i in range(len(TestX[0]))], axis=0)
    print ("done1")
    TestX2 = np.concatenate([TestY[0][i] for i in range(len(TestY[0]))], axis=0)
    print ("done2")
    TestX3 = np.concatenate([TestX[1][i] for i in range(len(TestX[0]))], axis=0)
    print ("done3")
    TestX4 = np.concatenate([TestY[1][i] for i in range(len(TestY[0]))], axis=0)
    print ("done4")

    TestX1 = TestX1.astype(np.float16)
    TestX2 = TestX2.astype(np.float16)

    TestX = np.concatenate((TestX1,TestX2), axis=0)
    TestY = np.concatenate((TestX3,TestX4), axis=0)

    print (TestX.shape)
    print (TestY.shape)

    TestX = TestX.astype(np.float16)
    test_X_train = np.transpose(TestX, (0, 3, 2, 1))
    test_X_train = test_X_train[:,:125,:,:]

    test_y_train = TestY.astype(np.int64)

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=True)

    return test_loader

def RPA_generator (patname, year, batch_size):

    test_X_train = np.load("/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/"+year+"/"+patname+"/totalx.npy")[:]
    test_y_train = np.load("/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/"+year+"/"+patname+"/totaly.npy")[:]

    test_X_train = test_X_train.astype(np.float16)

    # print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    # print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_y_train = test_y_train.astype(np.int64)

    test_X_train = np.transpose(test_X_train, (0, 2, 1, 3))

    print (test_X_train.shape)
    print (test_y_train.shape)

    # label_indices_to_delete = np.where(test_y_train == 1)[0]
    #
    # if len(label_indices_to_delete) > 20 and (year == "2014" or year == "2015"):
    #      indices_to_keep = np.random.choice(label_indices_to_delete, size=20, replace=False)
    #     indices_to_delete = np.setdiff1d(label_indices_to_delete, indices_to_keep)
    #      test_X_train = np.delete(test_X_train, indices_to_delete, axis=0)
    #     test_y_train = np.delete(test_y_train, indices_to_delete)

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def load_schizophrenia_data(healthy_dir, schizophrenia_dir):
    """
    Load the schizophrenia EEG dataset from the specified directories.
    
    Args:
        healthy_dir (str): Path to the directory containing healthy subject data
        schizophrenia_dir (str): Path to the directory containing schizophrenia subject data
        
    Returns:
        tuple: X (features) and y (labels) as numpy arrays
    """
    import os
    import numpy as np
    
    # Lists to store data and labels
    all_data = []
    labels = []
    
    # Load healthy subjects (label 0)
    for filename in os.listdir(healthy_dir):
        if filename.endswith('.eea'):
            file_path = os.path.join(healthy_dir, filename)
            
            # Load the raw data
            raw_data = np.loadtxt(file_path)
            
            # Reshape the data: Each subject has 16 channels with 7680 samples each
            # Shape becomes (16, 7680)
            subject_data = raw_data.reshape(16, 7680)
            
            all_data.append(subject_data)
            labels.append(0)  # 0 for healthy
    
    # Load schizophrenia subjects (label 1)
    for filename in os.listdir(schizophrenia_dir):
        if filename.endswith('.eea'):
            file_path = os.path.join(schizophrenia_dir, filename)
            
            # Load the raw data
            raw_data = np.loadtxt(file_path)
            
            # Reshape the data: Each subject has 16 channels with 7680 samples each
            subject_data = raw_data.reshape(16, 7680)
            
            all_data.append(subject_data)
            labels.append(1)  # 1 for schizophrenia
    
    # Convert to numpy arrays
    X = np.array(all_data)
    y = np.array(labels)
    
    return X, y

def preprocess_schizophrenia_data(X, segment_duration_seconds=12, sampling_rate=128):
    """
    Preprocess the schizophrenia EEG data.
    
    Args:
        X (numpy.ndarray): Raw EEG data with shape (n_subjects, n_channels, n_samples)
        segment_duration_seconds (int): Duration of each segment in seconds
        sampling_rate (int): Sampling rate of the EEG data
        
    Returns:
        numpy.ndarray: Preprocessed EEG data
    """
    import numpy as np
    from scipy import signal
    
    n_subjects, n_channels, n_samples = X.shape
    segment_samples = segment_duration_seconds * sampling_rate
    
    # Calculate how many full segments we can extract from each recording
    n_segments_per_subject = n_samples // segment_samples
    
    # Initialize the array for segmented data
    X_segmented = np.zeros((n_subjects * n_segments_per_subject, n_channels, segment_samples))
    
    # Segment the data
    for subject_idx in range(n_subjects):
        for segment_idx in range(n_segments_per_subject):
            start_idx = segment_idx * segment_samples
            end_idx = start_idx + segment_samples
            X_segmented[subject_idx * n_segments_per_subject + segment_idx] = X[subject_idx, :, start_idx:end_idx]
    
    # Apply bandpass filter (0.5-45 Hz) to remove noise and artifacts
    b, a = signal.butter(4, [0.5, 45], btype='bandpass', fs=sampling_rate)
    X_filtered = np.zeros_like(X_segmented)
    
    for i in range(X_segmented.shape[0]):
        for j in range(n_channels):
            X_filtered[i, j] = signal.filtfilt(b, a, X_segmented[i, j])
    
    # Apply STFT (Short-Time Fourier Transform) as done in the original code
    # This converts time-domain signals to time-frequency representation
    X_stft = np.zeros((X_filtered.shape[0], n_channels, 23, 125), dtype=np.complex64)
    
    for i in range(X_filtered.shape[0]):
        for j in range(n_channels):
            f, t, Zxx = signal.stft(X_filtered[i, j], fs=sampling_rate, nperseg=256, noverlap=128)
            # Extract relevant frequency components (up to ~45 Hz) 
            X_stft[i, j] = Zxx[:23, :125]
    
    # Take the magnitude of the STFT
    X_stft_mag = np.abs(X_stft)
    
    # Reshape to the format expected by the model: (n_samples, n_freq, n_time, n_channels)
    X_final = np.transpose(X_stft_mag, (0, 2, 3, 1))
    
    return X_final

def create_schizophrenia_datasets(healthy_dir='healthy', schizophrenia_dir='schiz', test_size=0.2, batch_size=64):
    """
    Create train and test datasets for the schizophrenia EEG data.
    
    Args:
        healthy_dir (str): Path to the directory containing healthy subject data
        schizophrenia_dir (str): Path to the directory containing schizophrenia subject data
        test_size (float): Proportion of data to use for testing
        batch_size (int): Batch size for the data loaders
        
    Returns:
        tuple: (train_loader, test_loader, n_classes)
    """
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    
    # Load and preprocess the data
    X_raw, y = load_schizophrenia_data(healthy_dir, schizophrenia_dir)
    X_processed = preprocess_schizophrenia_data(X_raw)
    
    # Reshape for the KAN model (flatten the 3D features)
    n_samples = X_processed.shape[0]
    X_reshaped = X_processed.reshape(n_samples, -1)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, 2  # 2 classes: healthy and schizophrenia