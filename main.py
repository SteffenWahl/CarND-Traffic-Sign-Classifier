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

# scale data
X_train = (X_train-128)/128
X_valid = (X_valid-128)/128
X_test = (X_test-128)/128

# reuse samples, such that all classes are equally often used
max_samples = np.max(trains_abs)
X_train_eq = np.empty(shape=[0, 32,32,3]) 
y_train_eq = np.array([])
for c in range(n_classes):
    valids = y_train==c
    v = np.array(range(len(valids)))
    v = v[valids]
    n_add = int(max_samples-valids.sum())
    randy = np.random.randint(len(v),size=n_add)
    np.random.randint(2, size=10)
    X_train_eq = np.append(X_train_eq, X_train[valids],axis=0)
    y_train_eq = np.append(y_train_eq, y_train[valids],axis=0)
    X_train_eq = np.append(X_train_eq, X_train[v[randy]],axis=0)
    y_train_eq = np.append(y_train_eq, y_train[v[randy]],axis=0)
X_train = X_train_eq
y_train = y_train_eq


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten


EPOCHS = 10
BATCH_SIZE = 128

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_shape[2], 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
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
    
    keep_prob = 0.5

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    fc1 = tf.nn.dropout(fc2, keep_prob)

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
x = tf.placeholder(tf.float32, (None, 32, 32, image_shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

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
        train_accuracy, res = evaluate_details(X_train, y_train)
        train_accuracy, res2 = evaluate_details(X_valid, y_valid)
        plt.figure(100+i)
        plt.plot(res,'b',res2,'r')
        plt.show(block=False)
        plt.pause(1)
        z = zip(range(len(res)),res)
        print('Train accuracy per class')
        for a,b in z:
            print(str(a)+'  '+str(b))

        print()
        
    saver.save(sess, './lenet')
    print("Model saved")



#with tf.Session() as sess:
#    saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#    test_accuracy = evaluate(X_test, y_test)
#    print("Test Accuracy = {:.3f}".format(test_accuracy))

plt.show()
