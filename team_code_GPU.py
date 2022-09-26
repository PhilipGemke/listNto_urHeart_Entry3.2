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
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tensorflow.config.experimental.set_memory_growth(device, True)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    ## Define lists age_dependent with split up in children
    neonate_percussive = list()
    infant_percussive = list()
    child_percussive = list()
    child_over120cm_percussive = list()
    child_under120cm_percussive = list()
    young_adult_percussive = list()
    adolescent_percussive = list()
    nan_percussive = list()

    neonate_normalized = list()
    infant_normalized = list()
    child_normalized = list()
    child_over120cm_normalized = list()
    child_under120cm_normalized = list()
    young_adult_normalized = list()
    adolescent_normalized = list()
    nan_normalized = list()

    neonate_murmurs = list()
    infant_murmurs = list()
    child_murmurs = list()
    child_over120cm_murmurs = list()
    child_under120cm_murmurs = list()
    young_adult_murmurs = list()
    adolescent_murmurs = list()
    nan_murmurs = list()

    neonate_outcomes = list()
    infant_outcomes = list()
    child_outcomes = list()
    child_over120cm_outcomes = list()
    child_under120cm_outcomes = list()
    young_adult_outcomes = list()
    adolescent_outcomes = list()
    nan_outcomes = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data, recordings and patient age
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)
        num_locations = get_num_locations(current_patient_data)
        current_patient_age = get_age(current_patient_data)

        # Extract features. 
        recording = choose_recording(current_patient_data, current_recordings)
        beats_normalized, beats_percussive = get_feature(recording)

        ## Extract labels with integer for 10 beats per patient
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            current_murmur = murmur_classes.index(murmur)

        ##Extract one-hot-encoding outcome for 10 beats per patient
        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1

        if current_patient_age == 'Neonate':
            for y in range(10):
                neonate_murmurs.append(current_murmur)
                neonate_outcomes.append(current_outcome)
                neonate_percussive.append(beats_percussive[y])
                neonate_normalized.append(beats_normalized[y])
        if current_patient_age == 'Infant':
            for y in range(10):
                infant_murmurs.append(current_murmur)
                infant_outcomes.append(current_outcome)
                infant_percussive.append(beats_percussive[y])
                infant_normalized.append(beats_normalized[y])
        if current_patient_age == 'Child':
            for y in range(10):
                child_murmurs.append(current_murmur)
                child_outcomes.append(current_outcome)
                child_percussive.append(beats_percussive[y])
                child_normalized.append(beats_normalized[y])
            current_patient_height = get_height(current_patient_data)
            if current_patient_height >120:
                for y in range(10):
                    child_over120cm_murmurs.append(current_murmur)
                    child_over120cm_outcomes.append(current_outcome)
                    child_over120cm_percussive.append(beats_percussive[y])
                    child_over120cm_normalized.append(beats_normalized[y])
            else:
                for y in range(10):
                    child_under120cm_murmurs.append(current_murmur)
                    child_under120cm_outcomes.append(current_outcome)
                    child_under120cm_percussive.append(beats_percussive[y])
                    child_under120cm_normalized.append(beats_normalized[y])
        if current_patient_age == 'Young adult':
            for y in range(10):
                young_adult_murmurs.append(current_murmur)
                young_adult_outcomes.append(current_outcome)
                young_adult_percussive.append(beats_percussive[y])
                young_adult_normalized.append(beats_normalized[y])
        if current_patient_age == 'Adolescent':
            for y in range(10):
                adolescent_murmurs.append(current_murmur)
                adolescent_outcomes.append(current_outcome)
                adolescent_percussive.append(beats_percussive[y])
                adolescent_normalized.append(beats_normalized[y])
        if current_patient_age == 'nan':
            for y in range(10):
                nan_murmurs.append(current_murmur)
                nan_outcomes.append(current_outcome)
                nan_percussive.append(beats_percussive[y])
                nan_normalized.append(beats_normalized[y])

    youngest_normalized = neonate_normalized + infant_normalized
    youngest_percussive = neonate_percussive + infant_percussive
    youngest_murmurs = neonate_murmurs + infant_murmurs
    youngest_outcomes = neonate_outcomes + infant_outcomes

    small_child_normalized = child_under120cm_normalized
    small_child_percussive = child_under120cm_percussive
    small_child_murmurs = child_under120cm_murmurs
    small_child_outcomes = child_under120cm_outcomes

    big_child_normalized = child_over120cm_normalized
    big_child_percussive = child_over120cm_percussive
    big_child_murmurs = child_over120cm_murmurs
    big_child_outcomes = child_over120cm_outcomes

    adult_normalized = young_adult_normalized + adolescent_normalized + nan_normalized
    adult_percussive = young_adult_percussive + adolescent_percussive + nan_percussive
    adult_murmurs = young_adult_murmurs + adolescent_murmurs + nan_murmurs
    adult_outcomes = young_adult_outcomes + adolescent_outcomes + nan_outcomes

    ##merge for feature extraction
    youngest_percussive = np.vstack(youngest_percussive)
    youngest_normalized = np.vstack(youngest_normalized)
    youngest_merge=[]
    for i in range(len(youngest_percussive)):
        youngest_merge.append(np.stack((youngest_percussive[i], youngest_normalized[i]), axis=1))
    youngest_merge = np.array(youngest_merge).astype('float32')

    small_child_percussive = np.vstack(small_child_percussive)
    small_child_normalized = np.vstack(small_child_normalized)
    small_child_merge=[]
    for i in range(len(small_child_percussive)):
        small_child_merge.append(np.stack((small_child_percussive[i], small_child_normalized[i]), axis=1))
    small_child_merge = np.array(small_child_merge).astype('float32')

    big_child_percussive = np.vstack(big_child_percussive)
    big_child_normalized = np.vstack(big_child_normalized)
    big_child_merge=[]
    for i in range(len(big_child_percussive)):
        big_child_merge.append(np.stack((big_child_percussive[i], big_child_normalized[i]), axis=1))
    big_child_merge = np.array(big_child_merge).astype('float32')

    adult_percussive = np.vstack(adult_percussive)
    adult_normalized = np.vstack(adult_normalized)
    adult_merge=[]
    for i in range(len(adult_percussive)):
        adult_merge.append(np.stack((adult_percussive[i], adult_normalized[i]), axis=1))
    adult_merge = np.array(adult_merge).astype('float32')

    youngest_murmurs = np.vstack(youngest_murmurs)
    youngest_outcomes = np.vstack(youngest_outcomes)
    small_child_murmurs = np.vstack(small_child_murmurs)
    small_child_outcomes = np.vstack(small_child_outcomes)
    big_child_murmurs = np.vstack(big_child_murmurs)
    big_child_outcomes = np.vstack(big_child_outcomes)
    adult_murmurs = np.vstack(adult_murmurs)
    adult_outcomes = np.vstack(adult_outcomes)

    ##shuffle all values
    np.random.seed(2)
    np.random.shuffle(youngest_murmurs)
    np.random.shuffle(youngest_outcomes)
    np.random.shuffle(youngest_merge)

    np.random.shuffle(small_child_murmurs)
    np.random.shuffle(small_child_outcomes)
    np.random.shuffle(small_child_merge)
    
    np.random.shuffle(big_child_murmurs)
    np.random.shuffle(big_child_outcomes)
    np.random.shuffle(big_child_merge)

    np.random.shuffle(adult_murmurs)
    np.random.shuffle(adult_outcomes)
    np.random.shuffle(adult_merge)

    # Train the model for murmur and outcome seperately and convert to json (to save models and weights in one data)
    if verbose >= 1:
        print('Training model...')

    print('youngest:', np.shape(youngest_merge), 'small_child:', np.shape(small_child_merge), 'big_child:',
    np.shape(big_child_merge), 'adult:', np.shape(adult_merge))


    youngest_murmur_model=Sequential()
    youngest_murmur_model.add(Bidirectional(LSTM(60, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    youngest_murmur_model.add(Bidirectional(LSTM(50, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    youngest_murmur_model.add(Bidirectional(LSTM(30, input_shape=(2400, 2))))
    youngest_murmur_model.add(Dense(3, activation='softmax'))
    youngest_murmur_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'], loss_weights=[1.,1.,0.25])
    youngest_murmur_model.fit(youngest_merge, youngest_murmurs, epochs=64, verbose=1, class_weight={0: 4.0, 1: 5.0, 2: 1.0})
    youngest_murmur_weights = youngest_murmur_model.get_weights()
    youngest_murmur_json = youngest_murmur_model.to_json()

    small_child_murmur_model=Sequential()
    small_child_murmur_model.add(Bidirectional(LSTM(50, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    small_child_murmur_model.add(Bidirectional(LSTM(50, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    small_child_murmur_model.add(Bidirectional(LSTM(30, input_shape=(2400, 2),dropout=0.1)))
    small_child_murmur_model.add(Dense(3, activation='softmax'))
    small_child_murmur_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'], loss_weights=[0.24,1.,0.08])
    small_child_murmur_model.fit(small_child_merge, small_child_murmurs, epochs=64, verbose=1, class_weight={0: 4.5, 1: 16.7, 2: 1.0})
    small_child_murmur_weights = small_child_murmur_model.get_weights()
    small_child_murmur_json = small_child_murmur_model.to_json()

    big_child_murmur_model=Sequential()
    big_child_murmur_model.add(Bidirectional(LSTM(80, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    big_child_murmur_model.add(Bidirectional(LSTM(20, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    big_child_murmur_model.add(Bidirectional(LSTM(20, input_shape=(2400, 2))))
    big_child_murmur_model.add(Dense(3, activation='softmax'))
    big_child_murmur_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'], loss_weights=[0.35,1.,0.07])
    big_child_murmur_model.fit(big_child_merge, big_child_murmurs, epochs=64, verbose=1, class_weight={0: 6.7, 1: 20.0, 2: 1.0})
    big_child_murmur_weights = big_child_murmur_model.get_weights()
    big_child_murmur_json = big_child_murmur_model.to_json()

    adult_murmur_model=Sequential()
    adult_murmur_model.add(Bidirectional(LSTM(40, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    adult_murmur_model.add(Bidirectional(LSTM(30, input_shape=(2400, 2), return_sequences=True, dropout=0.1)))
    adult_murmur_model.add(Bidirectional(LSTM(20, input_shape=(2400, 2))))
    adult_murmur_model.add(Dense(3, activation='softmax'))
    adult_murmur_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'], loss_weights=[0.21,1.,0.05])
    adult_murmur_model.fit(adult_merge, adult_murmurs, epochs=64, verbose=1, class_weight={0: 6.1, 1: 33.3, 2: 1.0})
    adult_murmur_weights = adult_murmur_model.get_weights()
    adult_murmur_json = adult_murmur_model.to_json()

    youngest_outcome_model=Sequential()
    youngest_outcome_model.add(Bidirectional(LSTM(70, input_shape=(2400, 2), return_sequences=True)))
    youngest_outcome_model.add(Bidirectional(LSTM(40, input_shape=(2400, 2), return_sequences=True)))
    youngest_outcome_model.add(Bidirectional(LSTM(40, input_shape=(2400, 2))))
    youngest_outcome_model.add(Dense(2, activation='softmax'))
    youngest_outcome_model.compile(loss='categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'], loss_weights=[0.79,1.])
    youngest_outcome_model.fit(youngest_merge, youngest_outcomes, epochs=1, verbose=1, class_weight={0: 1.0, 1: 1.6})
    youngest_outcome_weights = youngest_outcome_model.get_weights()
    youngest_outcome_json = youngest_outcome_model.to_json()

    small_child_outcome_model=Sequential()
    small_child_outcome_model.add(Bidirectional(LSTM(60, input_shape=(2400, 2), return_sequences=True)))
    small_child_outcome_model.add(Bidirectional(LSTM(60, input_shape=(2400, 2), return_sequences=True)))
    small_child_outcome_model.add(Bidirectional(LSTM(60, input_shape=(2400, 2))))
    small_child_outcome_model.add(Dense(2, activation='softmax'))
    small_child_outcome_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    small_child_outcome_model.fit(small_child_merge, small_child_outcomes, epochs=4, verbose=1)
    small_child_outcome_weights = small_child_outcome_model.get_weights()
    small_child_outcome_json = small_child_outcome_model.to_json()

    big_child_outcome_model=Sequential()
    big_child_outcome_model.add(Bidirectional(LSTM(70, input_shape=(2400, 2), return_sequences=True)))
    big_child_outcome_model.add(Bidirectional(LSTM(40, input_shape=(2400, 2), return_sequences=True)))
    big_child_outcome_model.add(Bidirectional(LSTM(20, input_shape=(2400, 2))))
    big_child_outcome_model.add(Dense(2, activation='softmax'))
    big_child_outcome_model.compile(loss='categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'], loss_weights=[1.0,0.79])
    big_child_outcome_model.fit(big_child_merge, big_child_outcomes, epochs=4, verbose=1, class_weight={0: 1.0, 1: 1.6})
    big_child_outcome_weights = big_child_outcome_model.get_weights()
    big_child_outcome_json = big_child_outcome_model.to_json()

    adult_outcome_model=Sequential()
    adult_outcome_model.add(Bidirectional(LSTM(80, input_shape=(2400, 2), return_sequences=True)))
    adult_outcome_model.add(Bidirectional(LSTM(50, input_shape=(2400, 2), return_sequences=True)))
    adult_outcome_model.add(Bidirectional(LSTM(40, input_shape=(2400, 2))))
    adult_outcome_model.add(Dense(2, activation='softmax'))
    adult_outcome_model.compile(loss='categorical_crossentropy', optimizer='adam', weighted_metrics=['acc'], loss_weights=[1., 0.37])
    adult_outcome_model.fit(adult_merge, adult_outcomes, epochs=4, verbose=1, class_weight={0: 2.7, 1: 1.0})
    adult_outcome_weights = adult_outcome_model.get_weights()
    adult_outcome_json = adult_outcome_model.to_json()
    #Save the model
    save_challenge_model(model_folder, youngest_murmur_json, youngest_murmur_weights, youngest_outcome_json, youngest_outcome_weights,
    small_child_murmur_json, small_child_murmur_weights, small_child_outcome_json, small_child_outcome_weights,
    big_child_murmur_json, big_child_murmur_weights, big_child_outcome_json, big_child_outcome_weights,
    adult_murmur_json, adult_murmur_weights, adult_outcome_json, adult_outcome_weights)



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
    #extract info from model.sav
    youngest_murmur_json = model['youngest_murmur_json']
    youngest_murmur_weights = model['youngest_murmur_weights']
    youngest_outcome_json = model['youngest_outcome_json']
    youngest_outcome_weights = model['youngest_outcome_weights']
    small_child_murmur_weights = model['small_child_murmur_weights']
    small_child_murmur_json = model['small_child_murmur_json']
    small_child_outcome_weights = model['small_child_outcome_weights']
    small_child_outcome_json = model['small_child_outcome_json']
    big_child_murmur_weights = model['big_child_murmur_weights']
    big_child_murmur_json = model['big_child_murmur_json']
    big_child_outcome_weights = model['big_child_outcome_weights']
    big_child_outcome_json = model['big_child_outcome_json']
    adult_murmur_weights = model['adult_murmur_weights']
    adult_murmur_json = model['adult_murmur_json']
    adult_outcome_weights = model['adult_outcome_weights']
    adult_outcome_json = model['adult_outcome_json']

    #define classes
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    #define probabilities
    murmur_probabilities = np.zeros(len(murmur_classes), dtype=np.int_)
    outcome_probabilities = np.zeros(len(outcome_classes), dtype=np.int_)

    ## rebuild models from json
    youngest_outcome_model = model_from_json(youngest_outcome_json)
    youngest_outcome_model.set_weights(youngest_outcome_weights)
    youngest_murmur_model = model_from_json(youngest_murmur_json)
    youngest_murmur_model.set_weights(youngest_murmur_weights)

    small_child_outcome_model = model_from_json(small_child_outcome_json)
    small_child_outcome_model.set_weights(small_child_outcome_weights)
    small_child_murmur_model = model_from_json(small_child_murmur_json)
    small_child_murmur_model.set_weights(small_child_murmur_weights)

    big_child_outcome_model = model_from_json(big_child_outcome_json)
    big_child_outcome_model.set_weights(big_child_outcome_weights)
    big_child_murmur_model = model_from_json(big_child_murmur_json)
    big_child_murmur_model.set_weights(big_child_murmur_weights)

    adult_outcome_model = model_from_json(adult_outcome_json)
    adult_outcome_model.set_weights(adult_outcome_weights)
    adult_murmur_model = model_from_json(adult_murmur_json)
    adult_murmur_model.set_weights(adult_murmur_weights)

    #predict probability of murmur or label for 10 beats of all locations and average probability
    num_locations = get_num_locations(data)
    current_patient_age = get_age(data)
    for i in range(num_locations):
        recording = recordings[i]
        beats_normalized, beats_percussive = get_feature(recording)
        features = np.array([beats_normalized, beats_percussive])
        for single_HB in range(10):
            if current_patient_age=='Neonate':
                current_murmur_probabilities = youngest_murmur_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                current_outcome_probabilities= youngest_outcome_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
            if current_patient_age=='Infant':
                current_murmur_probabilities = youngest_murmur_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                current_outcome_probabilities= youngest_outcome_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
            if current_patient_age=='Child':
                current_patient_height = get_height(data)
                if current_patient_height <121:
                    current_murmur_probabilities = small_child_murmur_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                    current_outcome_probabilities= small_child_outcome_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                else:
                    current_murmur_probabilities = big_child_murmur_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                    current_outcome_probabilities= big_child_outcome_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
            if current_patient_age=='Young adult':
                current_murmur_probabilities = adult_murmur_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                current_outcome_probabilities= adult_outcome_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
            if current_patient_age=='Adolescent':
                current_murmur_probabilities = adult_murmur_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                current_outcome_probabilities= adult_outcome_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
            if current_patient_age=='nan':
                current_murmur_probabilities = adult_murmur_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))
                current_outcome_probabilities= adult_outcome_model.predict(np.array([np.stack((beats_percussive[single_HB], beats_normalized[single_HB]), axis=1)]))

            murmur_probabilities = murmur_probabilities + current_murmur_probabilities
            outcome_probabilities = outcome_probabilities + current_outcome_probabilities
    murmur_probability = murmur_probabilities/(10*num_locations)
    murmur_probability = murmur_probability [0]
    outcome_probability = outcome_probabilities/(10*num_locations)
    outcome_probability = outcome_probability [0]


    # Calculate outcome_label using a tradeoff specifity --> sensitivity
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probability)
    murmur_labels[idx] = 1

    # Calculate outcome_label using a tradeoff specifity --> sensitivity
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    A_class = outcome_probability[:1]
    if np.max(A_class) > 0.48:
        idx = 0
    else:
        idx = np.argmax(outcome_probability)
    outcome_labels[idx] = 1


    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probability, outcome_probability))
    print(labels, probabilities)

    # Livemonitoring
    current_murmur = np.zeros(len(murmur_classes), dtype=int)
    murmur = get_murmur(data)
    if murmur in murmur_classes:
        j = murmur_classes.index(murmur)
        current_murmur[j] = 1
    current_outcome = np.zeros(len(outcome_classes), dtype=int)
    outcome = get_outcome(data)
    if outcome in outcome_classes:
        j = outcome_classes.index(outcome)
        current_outcome[j] = 1
    print(current_patient_age)
    if current_patient_age=='Child':
        print(get_height(data))
    num_locations = get_num_locations(data)
    print(current_murmur, current_outcome)


    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, youngest_murmur_json, youngest_murmur_weights, youngest_outcome_json, youngest_outcome_weights,
small_child_murmur_json, small_child_murmur_weights, small_child_outcome_json, small_child_outcome_weights,
big_child_murmur_json, big_child_murmur_weights, big_child_outcome_json, big_child_outcome_weights,
adult_murmur_json, adult_murmur_weights, adult_outcome_json, adult_outcome_weights):
    d = {'youngest_murmur_json': youngest_murmur_json, 'youngest_murmur_weights': youngest_murmur_weights,
    'youngest_outcome_json': youngest_outcome_json, 'youngest_outcome_weights': youngest_outcome_weights,
    'small_child_murmur_json': small_child_murmur_json, 'small_child_murmur_weights': small_child_murmur_weights,
    'small_child_outcome_json': small_child_outcome_json, 'small_child_outcome_weights': small_child_outcome_weights,
    'big_child_murmur_json': big_child_murmur_json, 'big_child_murmur_weights': big_child_murmur_weights,
    'big_child_outcome_json': big_child_outcome_json, 'big_child_outcome_weights': big_child_outcome_weights,
    'adult_murmur_json': adult_murmur_json, 'adult_murmur_weights': adult_murmur_weights ,
    'adult_outcome_json': adult_outcome_json, 'adult_outcome_weights': adult_outcome_weights}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)


# Create individual heart beats by using audio bpm algorithm and finding local peaks
# Return normalized heart beats and percussion filtered
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

    beats_normalized = list()
    for i in range(len(Peaks)):
        beat = np_normalized[Peaks[i]-1200:Peaks[i]+1200]
        if len(beat)==2400:
            beats_normalized.append(beat)
        else:
            beats_normalized.append(np_normalized[2000:4400])
    beats_normalized = np.array(beats_normalized)[std_percussion<np.mean(std_percussion)]
    beats_normalized = beats_normalized.tolist()
    beats_normalized.sort(key=ant.app_entropy)
    beats_normalized = np.array(beats_normalized)

    beats_percussive = list()
    for i in range(len(Peaks)):
        beat = recording_percussive[Peaks[i]-1200:Peaks[i]+1200]
        if len(beat)==2400:
            beats_percussive.append(beat)
        else:
            beats_percussive.append(recording_percussive[2000:4400])
    beats_percussive = np.array(beats_percussive)[std_percussion<np.mean(std_percussion)]
    beats_percussive = beats_percussive.tolist()
    beats_percussive.sort(key=ant.app_entropy)
    beats_percussive = np.array(beats_percussive)

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
                    for i in range(len(beats_normalized)):
                        repeated.append(beats_normalized[i])
                        repeated.append(beats_normalized[i])
                        percussive_repeated.append(beats_percussive[i])
                        percussive_repeated.append(beats_percussive[i])
            return np.array(repeated[:10]), np.array(percussive_repeated[:10]),
    else:
        print('nice beat detection')
        return beats_normalized[:10], beats_percussive[:10]
def choose_recording(data, recordings):
    locations=get_locations(data)
    number_location=len(locations)
    if get_murmur(data)=="Present":
        rec = recordings[locations.index(get_most_audible(data))]
    else:
        rec_list=[]
        for i in range(number_location):
            rec_list.append(recordings[i])
        rec_list.sort(key=len, reverse=True)
        rec = rec_list[0]

    return rec
def get_most_audible(data):
    most_audible = None
    for l in data.split('\n'):
        if l.startswith('#Most audible location:'):
            try:
                most_audible = l.split(': ')[1].strip()
            except:
                pass
    return most_audible
