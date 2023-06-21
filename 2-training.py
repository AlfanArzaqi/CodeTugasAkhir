import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout

# spesifikasi model CNN-LSTM
IMG_HEIGHT, IMG_WIDTH = 480, 640
SEQ_LENGTH = 20
MODEL_NAME = "SIBI.h5"
LABELS = ("Jumlah", "Kurang", "Sama Dengan")

def LoadDataset():
    total_class = len(LABELS)
    target_class = np.eye(total_class)
    
    # Definisikan path ke direktori data
    data_dir = os.getcwd() + "\\" + "Output"

    # Definisikan ukuran citra
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # Definisikan list untuk menyimpan data dan label
    data = []
    labels = []

    # Looping melalui setiap direktori dalam direktori data
    for (i, label) in enumerate(os.listdir(data_dir)):
        # Looping melalui setiap file citra dalam direktori label
        for sequence in os.listdir(os.path.join(data_dir, label)):
            sequence_dir = os.path.join(data_dir, label, sequence)
            frame_paths = sorted([os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir)])
            
            # Memastikan jumlah frame dalam setiap sequence sama
            if len(frame_paths) == SEQ_LENGTH:
                frames = []
                for frame_path in frame_paths:
                    # Load citra dan ubah ukuran
                    img = load_img(frame_path, target_size=IMG_SIZE)
                    # Ubah citra menjadi array NumPy
                    img_array = img_to_array(img)
                    # Normalisasi nilai piksel ke dalam rentang 0-1
                    img_array /= 255.0
                    frames.append(img_array)
                # Gabungkan beberapa frame menjadi satu sequence
                data.append(frames)
                labels.append(target_class[i])
                
    # Konversi list data dan label ke dalam array NumPy
    data = np.array(data)
    labels = np.array(labels)
    
    data = data.astype(np.float32)
    labels = labels.astype(np.float32)

    # Cetak bentuk data dan label
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    
    return data, labels

def CreateModel():
    # membuat model CNN-LSTM
    model = Sequential()

    # layer CNN 1
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'), input_shape=(SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    # layer CNN 2
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    # layer LSTM
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(len(LABELS), activation='softmax'))

    # kompilasi model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    
    return model

def TrainingModel():
    data, labels = LoadDataset()
    model = CreateModel()
    
    history = model.fit(data, labels, batch_size=16, epochs=10, shuffle=True)
    
    model.save(MODEL_NAME)
    
    return model, history

model, history = TrainingModel()

#c. Menampilkan Grafik Loss dan accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])

plt.title('model loss')
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.legend(['loss', 'accuracy'], loc='upper left')
plt.show()