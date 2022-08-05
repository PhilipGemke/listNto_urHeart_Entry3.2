 #!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
from helper_code import *
import numpy as np, os, tensorflow, json, joblib, pydub, librosa, presets, antropy as ant, codecs
from tensorflow import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from pydub import AudioSegment, effects
from presets import Preset #preset for open source music classification librosa
import librosa as _librosa #dummy import for preset


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):

    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    #show GPU state and adjuste settings
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    #os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tensorflow.config.experimental.set_memory_growth(device, True)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    percussive = list()
    raw = list()
    murmurs = list()
    outcomes = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
        num_locations = get_num_locations(current_patient_data)
        for location in range(num_locations):
            recording = current_recordings[location]
            beats_normalized, beats_percussive = get_feature(recording)
            percussive.append(beats_percussive)
            raw.append(beats_normalized)
            save_percussive = beats_percussive.tolist()
            save_normalized = beats_normalized.tolist()
            json.dump(save_percussive, codecs.open(os.path.join(model_folder, str(get_patient_id(current_patient_data))+ str(location) + 'save.json'), 'w', encoding='utf-8'),
                     separators=(',', ':'),
                      sort_keys=True,
                      indent=4)
            json.dump(save_normalized, codecs.open(os.path.join(model_folder, str(get_patient_id(current_patient_data))+ str(location) + str(9) + 'save.json'), 'w', encoding='utf-8'),
                     separators=(',', ':'),
                      sort_keys=True,
                      indent=4)
            #load_percussive = codecs.open(os.path.join(model_folder, str(get_patient_id(current_patient_data))+ str(location) + 'save.json'), 'r', encoding='utf-8').read()
            #beats_percussive = json.loads(load_percussive)
            #for single_HB in range(len(beats_percussive)):
            #    percussive.append(beats_percussive[single_HB])
            #load_raw = codecs.open(os.path.join(model_folder, str(get_patient_id(current_patient_data))+ str(location) + str(9) + 'save.json'), 'r', encoding='utf-8').read()
            #beats_raw = json.loads(load_raw)
            #for single_HB in range(len(beats_raw)):
            #    raw.append(beats_raw[single_HB])

        # Extract labels with integer for 10 beats per patient
        for location in range(num_locations):
            murmur = get_murmur(current_patient_data)
            if murmur in murmur_classes:
                j = murmur_classes.index(murmur)
            for y in range(10):
                murmurs.append(j)
        # Extract one-hot-encoding outcome for 10 beats per patient
        for location in range(num_locations):
            current_outcome = np.zeros(num_outcome_classes, dtype=int)
            outcome = get_outcome(current_patient_data)
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[j] = 1
            for y in range(10):
                outcomes.append(current_outcome)
    print(percussive[0][:10], raw[0][:10])
    merge=[]
    for i,j in zip(percussive,raw):
        for l in range(2400):
            merge.append((i[l],j[l]))
    merge = np.vstack(merge).reshape(len(percussive), 2400, 2).astype('float32')
    print(np.shape(merge), merge[0][:20])
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    # Train the model.
    if verbose >= 1:
        print('Training model...')
    ##Defining Bidirection LSTM
    murmur_model = make_murmur_model(merge, murmurs)
    murmur_weights = murmur_model.get_weights()
    murmur_json = murmur_model.to_json()
    outcome_model = make_outcome_model(merge, outcomes)
    outcome_weights = outcome_model.get_weights()
    outcome_json = outcome_model.to_json()
    #Save the model.
    save_challenge_model(model_folder, murmur_json, outcome_json, murmur_weights, outcome_weights)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    murmur_json = model['murmur_json']
    murmur_weights = model['murmur_weights']
    outcome_json = model['outcome_json']
    outcome_weights = model['outcome_weights']
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    #predict probability of murmur or label for 10 beats of all locations and average probability
    murmur_probabilities = np.zeros(len(murmur_classes), dtype=np.int_)
    outcome_probabilities = np.zeros(len(outcome_classes), dtype=np.int_)
    outcome_model = model_from_json(outcome_json)
    outcome_model.set_weights(outcome_weights)
    murmur_model = model_from_json(murmur_json)
    murmur_model.set_weights(murmur_weights)
    features = list()
    num_locations = get_num_locations(data)
    for i in range(num_locations):
        recording = recordings[i]
        beats_normalized, beats_percussive = get_feature(recording)
        features = features.reshape(10, 1, 2400, 2)
        for single_HB in range(10):
            current_murmur_probabilities = murmur_model.predict(features[single_HB
            current_outcome_probabilities= outcome_model.predict(features[single_HB])
            murmur_probabilities = murmur_probabilities + current_murmur_probabilities
            outcome_probabilities = outcome_probabilities + current_outcome_probabilities
    murmur_probability = murmur_probabilities/(10*num_locations)
    murmur_probability = murmur_probability [0]
    outcome_probability = outcome_probabilities/(10*num_locations)
    outcome_probability = outcome_probability [0]

    # Calculate outcome_label using a tradeoff specifity --> sensitivity
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    PU_class = murmur_probabilities[:2]
    print(PU_class)
    if PU_class.max() > 0.3:
        idx = np.argmax(PU_class)
    else:
        idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1

    # Calculate outcome_label using a tradeoff specifity --> sensitivity
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    A_class = outcome_probabilities[:1]
    if A_class.max() > 0.4:
        idx = np.argmax(A_class)
    else:
        idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1


    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcomdefaulte_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probability, outcome_probability))
    print(labels, probabilities)

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, murmur_json, outcome_json, murmur_weights, outcome_weights):
    d = {'murmur_json': murmur_json, 'outcome_json': outcome_json,
    'murmur_weights': murmur_weights, 'outcome_weights': outcome_weights}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


# Create individual heart beats by using audio bpm algorithm and finding local peaks
def get_feature(recording):

    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    unchanged_recording_in_segment_format = pydub.AudioSegment(recording.tobytes(), frame_rate=4000, sample_width=recording.dtype.itemsize, channels=1)
    dBFS_normalized = match_target_amplitude(unchanged_recording_in_segment_format, -20)
    array_normalized = dBFS_normalized.get_array_of_samples()
    np_normalized = np.array(array_normalized)
    np_normalized = np_normalized.astype(float)
    librosa = Preset(_librosa)
    librosa['sr'] = 4000
    librosa['hop_length'] = 32
    librosa['n_fft'] = 128
    librosa['win_length'] = 128
    stft = librosa.stft(np_normalized)
    harmonic_part, percussive_part = librosa.decompose.hpss(stft)
    recording_percussive = librosa.istft(percussive_part, length=len(np_normalized))
    _, Peaks = librosa.beat.beat_track(y=recording_percussive, tightness=128, units='samples', trim=False)
    beats_raw = list()
    for i in range(len(Peaks)):
        beat = recording[Peaks[i]-1200:Peaks[i]+1200]
        if len(beat)==2400:
            beats_raw.append(beat)
        else:
            beats_raw.append(recording[2000:4400])

    ##filter percussion with noise filter from raw data
    std_percussion = list()
    for i in range(len(beats_raw)):
        std_percussion.append(np.std(beats_raw[i]))
    #beats_raw.sort(key=ant.app_entropy)

    beats_normalized = list()
    for i in range(len(Peaks)):
        beat = recording[Peaks[i]-1200:Peaks[i]+1200]
        if len(beat)==2400:
            beats_normalized.append(beat)
        else:
            beats_normalized.append(recording[2000:4400])
    beats_normalized = np.array(beats_normalized)[std_percussion<np.mean(std_percussion)]
    beats_normalized = beats_normalized.tolist()
    beats_normalized.sort(key=ant.app_entropy)
    beats_normalized = np.array(beats_normalized)

    beats_percussive = list()
    for i in range(len(Peaks)):
        beat = recording[Peaks[i]-1200:Peaks[i]+1200]
        if len(beat)==2400:
            beats_percussive.append(beat)
        else:
            beats_percussive.append(recording[2000:4400])
    beats_percussive = np.array(beats_percussive)[std_percussion<np.mean(std_percussion)]
    beats_percussive = beats_percussive.tolist()
    beats_percussive.sort(key=ant.app_entropy)
    beats_percussive = np.array(beats_percussive)
    #std = list()
    #for i in range(len(beats_raw)):
    #    std.append(np.std(beats_raw[i]))
    #beats_raw = np.array(beats_raw)[std<np.mean(std)]
    if len(beats_normalized) < 11:
        if len(beats_normalized)==0:
            if len(recording_percussive) >24001:
                steps = range(0,24001,2400)
                normalized_exception = []
                percussive_exception = []
                for i in range(10):
                    normalized_exception.append(recording[steps[i]:steps[i+1]])
                    percussive_exception.append(recording_percussive[steps[i]:steps[i+1]])
                print('librosa beat detection did not work')
                return np.array(normalized_exception), np.array(percussive_exception)
            else:
                return np.array(np.tile(recording[2000:4400],10)), np.array(np.tile(recording_percussive[2000:4400],10))
                print('librosa beat detection did not work')
        else:
            print('had to repeat')
            repeated = []
            percussive_repeated = []
            for i in range(len(beats_normalized)):
                repeated.append(beats_normalized[i])
                repeated.append(beats_normalized[i])
                percussive_repeated.append(beats_percussive[i])
                percussive_repeated.append(beats_percussive[i])
                while len(repeated) < 11:
                    for i in range(len(beats_raw)):
                        repeated.append(beats_normalized[i])
                        repeated.append(beats_normalized[i])
                        percussive_repeated.append(beats_percussive[i])
                        percussive_repeated.append(beats_percussive[i])
            return np.array(repeated[:10]), np.array(percussive_repeated[:10]),
    else:
        print('nice beat detection')
        return beats_normalized[:10], beats_percussive[:10]


#Define Murmur LSTM Model with class weights and early stopping for cross validation
def make_murmur_model(X_train, y_train):
    verbose, epochs, batch_size = 1, 64, 0
    n_samples, n_features, n_outputs = X_train.shape[0], X_train.shape[1], 3


    murmur_model=Sequential()
    murmur_model.add(Bidirectional(LSTM(100, input_shape=(2400, 2))))
    murmur_model.add(Dense(n_outputs, activation='softmax'))
    murmur_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'])
    murmur_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return murmur_model

#Define Outcome LSTM Model with class weights and early stopping for cross validation
def make_outcome_model(X_train, y_train):
    verbose, epochs, batch_size = 1, 64, 0
    n_samples, n_features, n_timesteps, n_outputs = X_train.shape[0], 1, 2400, 2


    outcome_model=Sequential()
    outcome_model.add(Bidirectional(LSTM(80, input_shape=(2400, 2))))
    outcome_model.add(Dense(n_outputs, activation='softmax'))
    outcome_model.compile(loss='categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'])
    outcome_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return outcome_model
