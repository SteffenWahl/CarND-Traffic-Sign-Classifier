# Load pickled data
import pickle
import matplotlib.pyplot as plt
import numpy as np

training_file = '../data/train.p'
validation_file= '../data/valid.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_train_o = X_train
X_valid_o = X_valid
X_test_o = X_test
y_train_o = y_train
y_valid_o = y_valid
y_test_o = y_test

n_train = X_train.shape[0]

n_validation = X_valid.shape[0]

n_test = X_test.shape[0]

image_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])

n_classes = np.unique(y_train).size

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.

#show some random images
n_vis_imgs = [5,10]
fig, axes = plt.subplots(n_vis_imgs[0], n_vis_imgs[1], figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.3, wspace=0.05)
for ax, i in zip(axes.flat, range(n_vis_imgs[0]*n_vis_imgs[1])):
    randy = random.randint(0,n_train)

    ax.imshow(X_train[randy])

#show some stats
import csv
reader = csv.reader(open('signnames.csv', 'r'))
d = {}
for row in reader:
   k, v = row
   d[k] = v
print('Training data:')
print('|Train|Validate|Test|Name|')
trains = np.array([])
trains_abs = np.array([])
tests = np.array([])
valids = np.array([])
for class_id in range(n_classes):
    tr = (y_train==class_id).sum()
    va = (y_valid==class_id).sum()
    te = (y_test==class_id).sum()
    trains_abs = np.append(trains_abs,tr)
    trains = np.append(trains,tr/n_train)
    valids = np.append(valids,va/n_validation)
    tests =  np.append(tests,te/n_test)
    print(str(tr) + ' | ' + str(va) +' | ' +  str(te) + ' -> ' + d[str(class_id)] )
plt.figure(2)
plt.plot(trains,'r',valids,'g',tests,'b')
plt.show(block=False)
plt.pause(1)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle

#convert HSV
import cv2

def convert_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

X_o = np.zeros([X_train.shape[0],32,32,3])
for i in range(X_train.shape[0]):
     X_o[i] = convert_image(X_train[i])
X_train = X_o

X_o = np.zeros([X_valid.shape[0],32,32,3])
for i in range(X_valid.shape[0]):
     X_o[i] = convert_image(X_valid[i])
X_valid = X_o

# scale data
def scale(X):
    return (X-128.0)/128
def unscale(X):
    return (X*128.0)+128.0

X_train = scale(X_train)
X_valid = scale(X_valid)
X_test = scale(X_test)


# add samples for classes with problems...
n_add = (y_train==31).sum() - (y_train==21).sum()
valids = y_train==21
v = np.array(range(len(valids)))
v = v[valids]
randy = np.random.randint(len(v), size=n_add)
y_train = np.append(y_train, y_train[v[randy]], axis=0)
X_train = np.append(X_train, X_train[v[randy]], axis=0)

n_add = (y_train==5).sum() - (y_train==3).sum()
valids = y_train==3
v = np.array(range(len(valids)))
v = v[valids]
randy = np.random.randint(len(v), size=n_add)
y_train = np.append(y_train, y_train[v[randy]], axis=0)
X_train = np.append(X_train, X_train[v[randy]], axis=0)


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten


EPOCHS = 50
BATCH_SIZE = 128

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 18), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(18))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 18, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    keep_prob1 = 0.75
    fc1 = tf.nn.dropout(fc1, keep_prob1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    keep_prob2 = 0.75
    fc2 = tf.nn.dropout(fc2, keep_prob2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits   = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.0005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def evaluate_details(X_base, y_base):
    sess = tf.get_default_session()
    result = {}
    res = np.array([])
    for cid in range(n_classes):
        ids = y_base==cid
        if ids.sum() == 0:
            accuracy = float('nan')
        else:
            X_data = X_base[ids]
            y_data = y_base[ids]
            accuracy = evaluate(X_data,y_data)
        result[cid] = accuracy 
        res = np.append(res,accuracy)
    return result, res


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        print("EPOCH {} ...".format(i+1))
        train_accuracy = evaluate(X_train, y_train)
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        validation_accuracy = evaluate(X_valid, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#        train_accuracy, res = evaluate_details(X_train, y_train)
#        train_accuracy, res2 = evaluate_details(X_valid, y_valid)
#        plt.figure(100+i)
#        plt.plot(res,'b',res2,'r')
#        plt.show(block=False)
#        plt.pause(1)
#        z = zip(range(len(res)),res)
#        print('Train accuracy per class')
#        for a,b in z:
#            print(str(a)+'  '+str(b))
#
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")



#with tf.Session() as sess:
#    saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#    test_accuracy = evaluate(X_test, y_test)
#    print("Test Accuracy = {:.3f}".format(test_accuracy))



prediction = tf.argmax(logits, 1)
def evaluate_logits(X_data, y_data):
    num_examples = len(X_data)
    x_dat = np.array([])
    y_dat = np.array([])
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(prediction, feed_dict={x: batch_x})
        x_dat = np.append(x_dat,accuracy)
        y_dat = np.append(y_dat,batch_y)
    return x_dat, y_dat

soft_max = tf.nn.softmax(logits)
def evaluate_softmax(X_data):
    sess = tf.get_default_session()
    somax = sess.run(soft_max, feed_dict={x: X_data})
    return somax

# create a squared image from all the input images
def stich_images(imgs):
    shape = imgs.shape
    n_img = shape[0]
    ax_ = int(np.ceil(np.sqrt(n_img)))
    img = np.zeros((ax_*shape[1],ax_*shape[2],shape[3]))
    ctr1 = 0
    ctr2 = 0
    for i in range(n_img):
        img[(ctr1*shape[1]):((ctr1+1)*shape[1]),(ctr2*shape[2]):((ctr2+1)*shape[2]),:] = imgs[i]
        ctr1 += 1
        if ctr1 >= ax_:
            ctr1 = 0
            ctr2 += 1
    return img.astype('int')


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    a,b = evaluate_logits(X_train, y_train)

plt.figure(1000)
plt.hist(x=a, bins=43, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.figure(1001)
plt.hist(x=b[a!=b], bins=43, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.figure(1002)
plt.hist(x=a[a!=b], bins=43, color='#0504aa', alpha=0.7, rwidth=0.85)


defects = unscale(X_train[a!=b]).astype(int)

img = stich_images(defects)
plt.figure(10000)
plt.imshow(img.astype('int'))

plt.show()







own_file = './new_signs/data.p'

with open(own_file, mode='rb') as f:
    own = pickle.load(f)

X_own = own['X']
y_own = own['y']

#show images
plt.figure()
fig, axes = plt.subplots(1, 5,
                         subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.3, wspace=0.05)
axes.flat[0].imshow(X_own[0].astype('int'))
axes.flat[1].imshow(X_own[1].astype('int'))
axes.flat[2].imshow(X_own[2].astype('int'))
axes.flat[3].imshow(X_own[3].astype('int'))
axes.flat[4].imshow(X_own[4].astype('int'))

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

X_o = np.zeros(X_own.shape)
for i in range(X_own.shape[0]):
     X_o[i] = convert_image(X_own[i].astype('uint8'))
X_own = X_o

X_own = scale(X_own)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    pred_own,b = evaluate_logits(X_own, y_own)

accuracy = (y_own==pred_own).sum()/len(pred_own)


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    sm = evaluate_softmax(X_own)
    sm_red = sess.run(tf.nn.top_k(tf.constant(sm), k=5))

indices = sm_red.indices
vals = sm_red.values
for i in range(indices.shape[0]):
    indi = indices[i]
    vs = vals[i]
    print("Results for image " + str(i))
    for ii in range(indices.shape[1]):
        print("{0:.5f}".format(vs[ii])+" "+d[str(indi[ii])])



# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


sess = tf.InteractiveSession()
sess.as_default()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    outputFeatureMap(X_own[0],conv1_W,plt_num=1234)
