import numpy as np
import tensorflow as tf
import cv2
from src.trigconv2d import TrigConv2D

def grad_cam(model, image, class_index, layer_name = None):
    #Preprocess image
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis = 0)      #adds a batch dimension
    #Intermediate product is extracted
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (TrigConv2D, tf.keras.layers.Conv2D)):
                layer_name = layer.name
                break
    #new model creation
    gradient_model = tf.keras.Model(
        inputs = model.input,
        outputs = [model.get_layer(layer_name).output, model.output]
    )
    #set tape as we make forward pass
    with tf.GradientTape() as tape:
        conv_output, predictions = gradient_model(image)
        class_score = predictions[:, class_index]
    gradients = tape.gradient(class_score, conv_output)
    #Weight calculation
    weights = tf.reduce_mean(gradients, axis=(1,2))
    #Weighted Sum
    conv_output = conv_output[0]    #get rid of batch dimension for weights
    weights = weights[0]    #get rid of batch dimension for weights
    heatmap = tf.reduce_sum(conv_output * weights, axis = -1)    #takes weighted sum
    heatmap = tf.nn.relu(heatmap)    #GRAD CAM only works with magnitudes; negative values are already discarded by design
    heatmap = heatmap/(tf.reduce_max(heatmap) + 1e-10)    #normalize values, add very small positive number to denominator to ensure no division by 0
    heatmap = cv2.resize(heatmap.numpy(), (image.shape[2], image.shape[1]))

    return heatmap, predictions.numpy()[0]


def integrated_gradients(model, image, class_index, baseline = None, steps = 50):
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis = 0)

    if baseline is None:
        baseline = np.zeros_like(image)
    elif len(baseline.shape) == 3:
        baseline = np.expand_dims(baseline, axis = 0)
    image = tf.cast(image, tf.float32)
    baseline = tf.cast(baseline, tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)

    gradient_accumulator = tf.zeros_like(image)
    for alpha in alphas:
        interpolated = baseline + alpha * (image-baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = model(interpolated)
            target_score = predictions[:, class_index]
        gradients = tape.gradient(target_score, interpolated)
        gradient_accumulator+=gradients

    avg_gradient = gradient_accumulator/(steps+1)
    integrated_grads = (image - baseline) * avg_gradient
    final_predictions = model(image).numpy()[0]
    return integrated_grads.numpy()[0], final_predictions
