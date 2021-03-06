class XceptionTransfer():
  def __init__(self):
    import keras 
    import tensorflow as tf
    self.base_model = keras.applications.Xception(weights="imagenet",  input_shape=(150, 150, 3), include_top=False)  
    self.base_model.trainable = False

  def transferlearn(self, train_generator,validation_generator):
    import keras 
    import tensorflow as tf
    import pandas as pd
    import numpy as np 
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

    x = keras.Input(shape=(150, 150, 3))
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)
    x = self.base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  
    y = keras.layers.Dense(1)(x)

    self.model = keras.Model(x, y)
    self.model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.CategoricalAccuracy()])
    self.model.fit(train_generator, epochs= 20, validation_data= validation_generator)
    self.base_model.trainable = True
    self.model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.CategoricalAccuracy()])

    
    
    es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc= ModelCheckpoint('leafdata/xception_model.h5', monitor='val_loss',mode='min', verbose=1, save_best_only=True)
    cb_list=[es,mc]

    

    history = self.model.fit(train_generator, epochs= 30, steps_per_epoch=3, validation_data=validation_generator, validation_steps=3, callbacks=[cb_list])

  def evaluate(self, test_generator):
    from keras.models import load_model
    saved_model = load_model('leafdata/xception_model.h5')
    return saved_model.evaluate(test_generator)

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
        last_conv_layer = model.get_layer('conv2d_191')
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

    saved_model = load_model('leafdata/xception_model.h5')
    for item in os.listdir('leafdata/Predict'):
      path = os.path.join('leafdata/Predict', item)
      gradCAM(path, saved_model)
