from tensorflow.keras.layers import Input, Activation, MaxPooling2D, Conv2D, Dense, Flatten
from src.trigconv2d import TrigConv2D
from tensorflow.keras.models import Model

def create_trigconv2d_model(input_shape, num_classes):
  inputs = Input(shape=input_shape)   #defines input

  x = TrigConv2D(filters = 16, kernel_size=3, frequency=2.0)(inputs)  #applies this layer to the input tensor
  x = Activation('relu')(x)     #relu activation
  x = MaxPooling2D(pool_size=(2,2))(x)
  
  #add a convolution block
  x = Conv2D(32, (3,3), activation = 'relu')(x)
  x = MaxPooling2D(pool_size=(2,2))(x)
  
  #FC layers are introduced
  x = Flatten()(x)
  x  = Dense(64, activation='relu')(x)
  
  #define outputs
  outputs = Dense(num_classes, activation = 'softmax')(x)
  
  #create model
  return Model(inputs = inputs, outputs = outputs)
