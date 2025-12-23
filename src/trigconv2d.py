import numpy as np
import tensorflow as tf   #library we'll use to build the model itself
from tensorflow.keras.layers import Layer   #layer is the blueprint for all layers(Custom included)

class TrigConv2D(Layer):  #filters are learned through sin and cos functions #Layer lets it inherit all the standard features that any other keras layer has
    def __init__(self, filters, kernel_size, frequency = 1.0, **kwargs):   #filters determines depth of output # kernel_size determines size of kernel #frequency is freq of sin/cos function  #**kwargs basically means any other keyword arguments(so all other keras arguments are passed as well)
        super(TrigConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.frequency = frequency

    def build(self, input_shape):  #Keras automatically passes in shape of data, we want kernel depth to match inputs depth
        kernel_shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)   #kernel_size = (height, width, depth)
        kernels = []  #we're going to fill this with a bunch of kernels each which capture a different feature of the input, these are all combined in the singular output channel
        for i in range(self.filters):
            x = np.linspace(-1,1, self.kernel_size)   #creates an evenly spaced grid with kernel_size number of elements between -1 and 1
            y = np.linspace(-1,1, self.kernel_size)
            x_grid, y_grid = np.meshgrid(x,y)    #takes the two 1-D arrays and combines it into a 2d matrix
            if i % 2 == 0:       #we're building the number of kernels equal to filters, every even numbered kernel is sin and every odd numbered kernel is cos
                kernel = np.sin(self.frequency * (x_grid + y_grid))
            else:
                kernel = np.cos(self.frequency * (x_grid + y_grid))  #we generate 2d wave pattern, capturing variance across x and y dimension
            #take raw sin and cos and put it in 
            kernel = kernel[:, :, np.newaxis, np.newaxis]  #we add two new dimensions to the kernel, an input channel and output channel
            kernel = np.repeat(kernel, input_shape[-1], axis = 2)    #copies kernel dimensions across all input channels
            kernels.append(kernel)  #appends the kernel we just made to the kernel list
        self.kernel = tf.constant(np.concatenate(kernels, axis = 3), dtype = tf.float32)  #constant means no trained weights, the concatenate puts the different kernels in kernels[] together on the axis of the output channel, which is one
        super(TrigConv2D, self).build(input_shape) #calls the Keras parent function to build a layer with the specified input shape
        
    def call(self, inputs):    #defines what layer does with inputs
        return tf.nn.conv2d(inputs, self.kernel, strides = [1,1,1,1], padding = 'SAME') #inputs is the data going in, these are the combination of filters, strides tells the kernel
                                                                                     #how to move across the data, padding = "SAME" keeps the image size the same beacuse edges
                                                                                     #are weird
