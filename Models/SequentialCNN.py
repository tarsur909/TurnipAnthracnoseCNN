class SequentialCNN():
  def __init__(self):
    import keras 
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D



    self.model = Sequential()
    self.model.add(Conv2D(filters=16, kernel_size=2, input_shape=(150, 150, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=2))
    self.model.add(Dropout(0.2))

    self.model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    self.model.add(MaxPooling2D(pool_size=2))
    self.model.add(Dropout(0.2))

    self.model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    self.model.add(MaxPooling2D(pool_size=2))
    self.model.add(Dropout(0.2))

    self.model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    self.model.add(MaxPooling2D(pool_size=2))
    self.model.add(Dropout(0.2))

    self.model.add(GlobalAveragePooling2D())

    self.model.add(Dense(3, activation='softmax')) 

    self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 

  def fit(self, train_generator, validation_generator):
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
    es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc= ModelCheckpoint('leafdata/sequential_model.h5', monitor='val_loss',mode='min', verbose=1, save_best_only=True)
    cb_list=[es,mc]
    history = self.model.fit(train_generator, epochs= 30, steps_per_epoch=3, validation_data=validation_generator, validation_steps=3, callbacks=[cb_list])

  def evaluate(self, test_generator):
    from keras.models import load_model
    saved_model = load_model('leafdata/sequential_model.h5')
    return saved_model.evaluate(test_generator)

  def predict(self):
      def predict(self):
        from keras.models import load_model
        import tensorflow as tf
        import os
        import numpy as np
        import tensorflow.keras.backend as K
        from tensorflow.keras.preprocessing import image
        from tensorflow.python.keras.applications.xception import preprocess_input, decode_predictions

    def gradCAM(path, model):
      intensity = 0.5
      res = 250

      img = image.load_img(path, target_size=(150, 150))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)

      preds = model.predict(x)
      print(decode_predictions(preds)[0][0][1]) 

      with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_4')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
      heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
      heatmap = np.maximum(heatmap, 0)
      heatmap /= np.max(heatmap)
      heatmap = heatmap.reshape((8, 8))

      img = cv2.imread(path)

      heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

      heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

      img = heatmap * intensity + img

      cv2_imshow(cv2.resize(cv2.imread(path), (res, res)))
      cv2_imshow(cv2.resize(img, (res, res)))

    saved_model = load_model('leafdata/sequential_model.h5')
    for item in os.listdir('leafdata/Predict'):
      path = os.path.join('leafdata/Predict', item)
      gradCAM(path, saved_model)
