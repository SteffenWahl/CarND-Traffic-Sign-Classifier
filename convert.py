import numpy as np
import cv2
import pickle
files = ['new_signs/80.jpg', 'new_signs/aufhebung.jpg', 'new_signs/einfahrt_verboten.jpg', 'new_signs/stop.jpg', 'new_signs/vorfahrt.jpg']

imgs = np.zeros((5,32,32,3))
for idx in range(len(files)):
    img = cv2.imread(files[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('int')
    imgs[idx,:,:,:] = img

labels = [5,32,15,14,12]

data = {"X":imgs,"y":labels}

with open('./own_data.p', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
