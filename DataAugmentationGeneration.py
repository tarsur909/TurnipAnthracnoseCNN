class DataAugmentationGeneration():
  def __init__(self):
    from tensorflow.python.keras.applications.xception import preprocess_input
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    self.data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip = True, rotation_range = 360, featurewise_center = True, shear_range=0.2, zoom_range=0.2,featurewise_std_normalization = True)

  def train(self):
    train_generator = self.data_generator.flow_from_directory('leafdata/Training', target_size=(150, 150), batch_size=12, shuffle = "True", class_mode='categorical')
    return train_generator

  def validation(self):
    validation_generator = self.data_generator.flow_from_directory('leafdata/Validation', target_size=(150, 150), batch_size=12, shuffle = "True", class_mode='categorical')
    return validation_generator

  def test(self):
    test_generator = self.data_generator.flow_from_directory('leafdata/Testing', target_size=(150, 150), batch_size=12, class_mode='categorical')
    return test_generator
