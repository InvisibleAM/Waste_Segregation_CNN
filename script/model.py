model=Sequential()


model.add(Conv2D(32,(3,3), padding='same',input_shape=(300,300,3),activation='relu'))
model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))


model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))


model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))


model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))

filepath="trained_model.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]
