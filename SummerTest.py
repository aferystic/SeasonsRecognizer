import numpy as np
import glob
from keras.preprocessing import image
from keras.models import load_model

model = load_model('./seasonsAIPostTrain.h5')

images = []
for filename in glob.glob(r'C:\Users\Aneta\PycharmProjects\biai\training_set\summer\*.*'):
    im = image.load_img(filename, target_size = (64,64))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    images.append(im)

OK = 0
Count0 = 0
Count1 = 0
Count3 = 0
NotOK =0


for image in images:
    result = model.predict(image)
    prediction = result.argmax(axis=-1)
    if prediction == 2:
        OK=OK+1
    elif prediction ==0:
        Count0 = Count0+1
        NotOK=NotOK+1
    elif prediction == 1:
        Count1 = Count1 + 1
        NotOK=NotOK+1
    else:
        Count3=Count3+1
        NotOK=NotOK+1

print(r'Summer accuracy: {0}'.format(OK/(OK+NotOK)))
print(r'Spring accuracy: {0}'.format(Count1/(OK+NotOK)))
print(r'Autumn accuracy: {0}'.format(Count0/(OK+NotOK)))
print(r'Winter accuracy: {0}'.format(Count3/(OK+NotOK)))