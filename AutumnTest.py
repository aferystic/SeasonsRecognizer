import numpy as np
import glob
from keras.preprocessing import image
from keras.models import load_model

model = load_model('./seasonsAIPostTrain.h5')

images = []
for filename in glob.glob(r'C:\Users\Aneta\PycharmProjects\biai\training_set\autumn\*.*'):
    im = image.load_img(filename, target_size = (64,64))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    images.append(im)

OK = 0
NotOK = 0

for image in images:
    result = model.predict(image)
    prediction = result.argmax(axis=-1)
    if prediction == 0:
        OK=OK+1
    else:
        NotOK=NotOK+1

print(r'Autumn accuracy: {0}'.format(OK/(OK+NotOK)))


