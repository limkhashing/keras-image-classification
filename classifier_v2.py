import numpy as np

from keras.preprocessing import image
from keras.models import load_model

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

# how do I know which class name in each index?
class_names = ["Driving License", "Identity Card"]

# model v2 is binary classifier
# {'driving_license': 0, 'identity_card': 1}
model = load_model('./model/model_v2.h5')

# load the image
test_image = image.load_img('./dataset_v2/single_prediction/test 2.jpg', target_size = INPUT_SHAPE)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

#####
# Don't have probabilities because the model is not probabilistic
# it's either 0 or 1, taking 0.5 as the activation line
# https://stackoverflow.com/questions/66704712/keras-image-binary-classification-which-class-is-assigned-probability-0-and-1
# https://stackoverflow.com/questions/52018645/how-do-i-determine-the-binary-class-predicted-by-a-convolutional-neural-network
#####
# probability = model.predict(test_image)
# prediction = np.argmax(probability[0])
# print("Probability: ", probability)
# print("Prediction: ", prediction)
# print("Prediction class: ", class_names[int(prediction)])

predictions = model.predict(test_image)
print("Prediction: ", predictions)
print("Predicted class: ", class_names[int(predictions)])