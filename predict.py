import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np

model=load_model("model2.h5")

def load_image(img_path, show=True):
    img_original = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_original)                           
        plt.axis('off')
        plt.show()
    return img_tensor

new_image = load_image("sp (3).jpg")
pred = model.predict(new_image)
print("CRAB : " +str(pred[0][0]))
print("LOBSTER : " +str(pred[0][1]))
print("OCTOPUS : " +str(pred[0][2]))
print("SCORPION : " +str(pred[0][3]))
print("SPIDER : " +str(pred[0][4]))
