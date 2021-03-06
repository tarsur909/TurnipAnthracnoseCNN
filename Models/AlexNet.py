class AlexNet():
  def __init__(self):
    from tensorflow import keras
    self.model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    ])

  def fit(self, train_generator, validation_generator):
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
    es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc= ModelCheckpoint('leafdata/alexnet_model.h5', monitor='val_loss',mode='min', verbose=1, save_best_only=True)
    cb_list=[es,mc]
    history = self.model.fit(train_generator, epochs= 30, steps_per_epoch=3, validation_data=validation_generator, validation_steps=3, callbacks=[cb_list])

  def evaluate(self, test_generator):
    from keras.models import load_model
    saved_model = load_model('leafdata/alexnet_model.h5')
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

      img = image.load_img(path, target_size=(227, 227))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)

      preds = model.predict(x)
      print(decode_predictions(preds)[0][0][1]) 

      with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_5')
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

    saved_model = load_model('leafdata/alexnet_model.h5')
    for item in os.listdir('leafdata/Predict'):
      path = os.path.join('leafdata/Predict', item)
      gradCAM(path, saved_model)
