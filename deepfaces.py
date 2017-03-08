
# coding: utf-8

# # Part 10

# # Import Packages

## Note: The following code is written in python 2 and with tensorflow 1.0 in
## Anaconda distributions

## File "bvlc_alexnet.npy" need to be under the same directory as this file

# In[3]:

from numpy import *
import glob
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import csv
import pandas as pd
import tensorflow as tf
import random
import cPickle


# # Import Data

# In[4]:

def get_all_images(paths):
    image_list = glob.glob(paths)
    return image_list


# In[5]:

def crop_image(image_path,crop_coordinates, im_size_x, im_size_y):  
    image = imread(image_path)
    image_name = image_path.split('/')[-1]
    x1 = int(crop_coordinates[image_name].split(',')[0])
    y1 = int(crop_coordinates[image_name].split(',')[1])
    x2 = int(crop_coordinates[image_name].split(',')[2])
    y2 = int(crop_coordinates[image_name].split(',')[3])
    
    cropped  = image[y1:y2, x1:x2]
    std_cropped = imresize(cropped, (im_size_x, im_size_y))
    #std_gray_cropped = rgb2gray(std_cropped)
    imshow(std_cropped)

    return [image_name, std_cropped]


# In[6]:

def update_images(image_list, crop_coordinates,  im_size_h, im_size_v):
    if not os.path.exists(os.getcwd() + '/cropped2'):
        os.makedirs(os.getcwd() + '/cropped2')
    else:
        pass
    crop_path = os.getcwd() + '/cropped2/'

    for i in image_list:
        try:
                plt.show()
                image_name, std_cropped = crop_image(i,crop_coordinates, im_size_h, im_size_v)
                imsave(crop_path + image_name, std_cropped)
        except:
            pass     
        
    print ("image update completed")
    
    return None


# ### Execute update images function 

# In[7]:

reader = csv.reader(open(os.getcwd() + '/Namelist/crop coordinates.csv', 'r'))
crop_coordinates = {}
for row in reader:
    coordinate,image_name = row
    crop_coordinates[image_name] = coordinate

reader = csv.reader(open(os.getcwd() + '/Namelist/hash value.csv', 'r'))
hash_values = {}
for row in reader:
    hash_value,image_name = row
    hash_values[image_name] = hash_value

image_list_uncropped = get_all_images(os.getcwd() + "/uncropped/*")
#remove_bad_images(image_list_uncropped, hash_values)
im_size_x = 227
im_size_y = 227 
image_list_updated = get_all_images(os.getcwd() + "/uncropped/*")
#update_images(image_list_updated,crop_coordinates, im_size_x, im_size_y)


# In[8]:

def clear_non_readable_files(actor_name_dataframe,image_list):
    image_name_list = []
    for i in image_list:
        image_name = i.split('/')[-1]
        image_name_list.append(image_name)
    actor_dataframe = actor_name_dataframe[actor_name_dataframe['image_name'].isin(image_name_list)]    
    return [image_name_list, actor_dataframe]


# In[9]:

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 6))


# ## Build training, validation and test sets

# ### build actor name dataframes

# In[10]:

uncropped_path = os.getcwd() + "/uncropped/*"
cropped_path = os.getcwd() + "/cropped2/*"
act = list(set([a.split("\t")[0] for a in open(os.getcwd() + "/faces_subset.txt").readlines()]))
image_list = get_all_images(uncropped_path)


# In[11]:

name_dict = {}
for i in act:
    last_name = (i.split(' ')[1]).lower()
    for j in glob.glob(os.getcwd() + "/cropped/" + last_name + "*"):
        image_name = j.split("/")[-1]
        name_dict[image_name] = i

actor_name_dataframe_origin = pd.DataFrame.from_dict(name_dict,orient='index')
actor_name_dataframe_origin['image_name'] = actor_name_dataframe_origin.index
actor_name_dataframe_origin = actor_name_dataframe_origin.rename(columns = {0: 'actor_names'})


actor_dataframe = clear_non_readable_files(actor_name_dataframe_origin, image_list)[1]
image_name_list = clear_non_readable_files(actor_name_dataframe_origin, image_list)[0]


# ### Build training, validation and test sets for a single actor

# In[12]:

def separate_training_sets(actor_name,actor_name_dataframe, image_name_list, cropped_path):
    
    image_paths = {}
    for i in image_name_list:
        image_paths[i] = cropped_path[:-1] + i 
    #image_paths_col = pd.Series(image_paths)
    image_paths_col = actor_name_dataframe['image_name'].map(image_paths).to_frame()
    image_paths_col.columns  = ["image_path"]
    actor_name_dataframe = actor_name_dataframe.join(image_paths_col)
    actor_image_df = actor_name_dataframe[actor_name_dataframe['actor_names'] == actor_name]
    
    test_set = actor_image_df.sample(n = 30, random_state=1)
    train_validation = actor_image_df.drop(test_set.index)
    validation_set = train_validation.sample(n = 10, random_state=1)
    training_set =  train_validation.drop(validation_set.index)

    return [training_set, test_set, validation_set]


# In[13]:

def organize_all_data(actor_list, actor_name_dataframe, image_name_list, cropped_path):
    organized_data = {}
    for i in actor_list:
        organized_data[i] = separate_training_sets(i,actor_name_dataframe, image_name_list, cropped_path)
        
    return organized_data


# In[14]:

def one_hot_encode_set(df):
    encoded_cols =  pd.get_dummies(df['actor_names']) 
    one_hot_full_data = df.join(encoded_cols)    
    return one_hot_full_data


# In[15]:

organized_data = organize_all_data(act, actor_dataframe, image_name_list, cropped_path)


# In[16]:

act6 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
training_df = [organized_data[i][0] for i in act6]
test_df = [organized_data[i][1] for i in act6]
validation_df = [organized_data[i][2] for i in act6]
training_df = one_hot_encode_set(pd.concat(training_df))
test_df = one_hot_encode_set(pd.concat(test_df))
validation_df = one_hot_encode_set(pd.concat(validation_df))


# In[17]:

def get_convolution_image_list(training_df):
    train_list = []
    for image_path in training_df['image_path']:
        image =(imread(image_path)[:,:,:3]).astype(float32)
        image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
        train_list.append(image)
    return train_list


# In[18]:

train_list = np.array(get_convolution_image_list(training_df))


# ###  Build training, validation and test sets for all actors in forms of dataframes

# ## Test Images

# In[19]:

net_data = load("bvlc_alexnet.npy").item()


# # Add Convolution Layer

# In[20]:

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


# # convolution layer 1
# ### conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')

# In[21]:

xdim = (227,227, 3)
input_x = tf.placeholder(tf.float32, (None,) + xdim)


# In[22]:

filter_height = 11; filter_width = 11; num_feature_maps = 96; stride_height = 4; stride_width = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(input_x, conv1W, conv1b, filter_height, filter_width, num_feature_maps, stride_height, stride_width, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)


# ## local response normalization 1

# In[23]:

radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)


# # Max Pooling Layer 1
# ## max_pool(3, 3, 2, 2, padding='VALID', name='pool1')

# In[24]:

filter_height = 3; filter_width = 3; stride_height = 2; stride_width = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_height, stride_width, 1], padding=padding)


# # Convolution Layer 2
# ## conv(5, 5, 256, 1, 1, group=2, name='conv2')

# In[25]:

filter_height = 5; filter_width = 5; num_feature_maps = 256; stride_height = 1; stride_width = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, filter_height, filter_width, num_feature_maps, stride_height, stride_width, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


# # local response normalization 2

# In[26]:

radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)


# # Max Pool Layer 2

# In[27]:

filter_height = 3; filter_width = 3; stride_height = 2; stride_width = 2; padding = 'VALID'
maxpool2 =tf.nn.max_pool(lrn2, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_height, stride_width, 1], padding=padding)


# # Convolution Layer 3
# ## conv(3, 3, 384, 1, 1, name='conv3')

# In[28]:

filter_height = 3; filter_width = 3; num_feature_maps = 384; stride_height = 1; stride_width = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, filter_height, filter_width, num_feature_maps, stride_height, stride_width, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)


# # Convolution Layer 4
# ## conv(3, 3, 384, 1, 1, group=2, name='conv4')

# In[29]:

filter_height = 3; filter_width = 3; num_feature_maps = 384; stride_height = 1; stride_width = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, filter_height, filter_width, num_feature_maps, stride_height, stride_width, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


# # Extract the 4th Activation Units as features

# In[30]:

def get_test_set_parameters_with_AlexNet(test_df, input_x, AN_layer): 
    
    test_list = np.array(get_convolution_image_list(test_df))

    test_x = sess.run(AN_layer, feed_dict = {input_x: test_list})
    
    ycols = test_df.columns.values.tolist()[-6:]
    test_y = test_df[ycols].as_matrix()
    
    return [test_x, test_y]


# ## Neural Network Training

# ### Initialize x, y, W and b

# In[31]:

xdim =  tuple(conv4.get_shape().as_list()[1:])
x = tf.placeholder(tf.float32, (None,) + xdim)
y_predict = tf.placeholder(tf.float32, [None, 6])


# In[32]:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

test_x, test_y =  get_test_set_parameters_with_AlexNet(test_df, input_x, conv4)
validation_x, validation_y = get_test_set_parameters_with_AlexNet(validation_df, input_x, conv4)
full_train_x, full_train_y = get_test_set_parameters_with_AlexNet(training_df, input_x, conv4)


# ### Define layers

# In[33]:

num_hidden_units = 400

# weights for the edges from input to the first hidden layer, initalized by random normal
W0 = tf.Variable(tf.random_normal([int(tf.reshape(x, [-1, int(prod(x.get_shape()[1:]))]).get_shape()[1]), num_hidden_units], stddev=0.00001, seed = 1)/1000000)
b0 = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.00001 , seed = 1)/10000000)

# weights for the edges from the first hidden layer to output
W1 = tf.Variable(tf.random_normal([num_hidden_units, 6], stddev=0.00001, seed = 1)/1000000)
b1 = tf.Variable(tf.random_normal([6], stddev=0.00001, seed = 1)/1000000)


# In[34]:

layer1 = tf.nn.relu_layer(tf.reshape(x, [-1, int(prod(x.shape[1:]))]), W0, b0)
layer2 = tf.matmul(layer1, W1)+b1
y_true = tf.nn.softmax(layer2)

correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ### Cost functions and train the nerual network

# In[35]:

lamda = 0.01
large_w_penalty =lamda*tf.reduce_sum(tf.square(W0))+lamda*tf.reduce_sum(tf.square(W1))
log_likelihood_cost = -tf.reduce_sum(y_predict*tf.log(y_true))+large_w_penalty
train_step = tf.train.AdamOptimizer(0.0001).minimize(log_likelihood_cost)


# In[36]:

def train_neural_network(test_x, test_y, validation_x, validation_y, full_train_x, full_train_y):
    train_learning_curve = {}
    test_learning_curve = {}
    validation_learning_curve = {}
    for i in range(101):
        np.random.seed(i+1)
        idx = array(np.random.permutation(len(full_train_x))[:12])
        batch_x =  array(full_train_x)[idx]
        batch_y = array(full_train_y)[idx]
        sess.run(train_step, feed_dict={x: batch_x, y_predict: batch_y})
        
        if i % 1 == 0 :
                print "i=",i

                print "Train:", str(sess.run(accuracy, feed_dict={x: full_train_x, y_predict: full_train_y})*100)+"%"
                train_learning_curve [i] = sess.run(accuracy, feed_dict={x: full_train_x, y_predict: full_train_y})*100
                print "Test:", str(sess.run(accuracy, feed_dict={x: test_x, y_predict: test_y}) *100) + "%"
                test_learning_curve[i] = sess.run(accuracy, feed_dict={x: test_x, y_predict: test_y})*100
                print "Validation:", str(sess.run(accuracy, feed_dict={x: validation_x, y_predict: validation_y}) *100) + "%"
                validation_learning_curve[i] = sess.run(accuracy, feed_dict={x: validation_x, y_predict: validation_y})*100
                print "Penalty:", sess.run(large_w_penalty)
                print "Cost Function Value:", sess.run(log_likelihood_cost, feed_dict={x: batch_x, y_predict: batch_y}) 
                print "\n"
                snapshot = {}
                snapshot["W0"] = sess.run(W0)
                snapshot["W1"] = sess.run(W1)
                snapshot["b0"] = sess.run(b0)
                snapshot["b1"] = sess.run(b1)
#         if i % 50 == 0:
#                 if not os.path.exists(os.getcwd() + '/weights_AN'):
#                         os.makedirs(os.getcwd() + '/weights_AN')
#                 else:
#                          pass
#                 cPickle.dump(snapshot,  open("weights_AN/AlexNet_weight"+str(i)+".pkl", "w"))
               
    learning_curves = [train_learning_curve, test_learning_curve, validation_learning_curve]           
    return [snapshot, learning_curves]


# In[37]:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
snapshot, learning_curves  =  train_neural_network(test_x, test_y, validation_x, validation_y, full_train_x, full_train_y)


# In[38]:

def plot_learning_curves(learning_curve):
    
    if not os.path.exists(os.getcwd() + '/results'):
         os.makedirs(os.getcwd() + '/results')
    else: 
        pass 
    training_performances, test_performances, validation_performances = learning_curve
    training_list = sorted(training_performances.items())
    test_list = sorted(test_performances.items())
    validation_list = sorted(validation_performances.items())
    n, accuracy_training = zip(*training_list)
    n, accuracy_test = zip(*test_list)
    n, accuracy_validation = zip(*validation_list)
    plt.figure(figsize=(10,6))
    plt.title('Accuracy of Training Set, Test Set and Validation Set')
    plt.xlabel('Number of Iterations n')
    plt.ylabel('Accuracy %')
    
    plt.plot(n, accuracy_training, label = 'Training Set Performances')
    plt.plot(n, accuracy_test, label = 'Test Set Performances')
    plt.plot(n, accuracy_validation, label = 'Validation Set Performances')
    plt.legend(loc = 'bottom right')
    figure = plt.gcf()
    plt.show()
    figure.savefig(os.getcwd() + '/results/Part 10 - AlexNet Neural Network Learning Curves.png')
    
    return None


# In[39]:

plot_learning_curves(learning_curves)


# # Bonus
# ## AlexNet Convolution Layer 4 Visualizations

# In[40]:

conv4_feature_maps = sess.run(conv4, feed_dict = {input_x: train_list})


# In[41]:

def plot_feature_map_at_different_locations(conv4_feature_maps):
    
    if not os.path.exists(os.getcwd() + '/results'):
         os.makedirs(os.getcwd() + '/results')
    else: 
        pass 
    feature_maps, axis = plt.subplots(3, 3, figsize=(20,20))
    for i in range(3):
        for j in range(3):
            feature_map = np.asarray(conv4_feature_maps[j,:,:, i*2])
            axis[i, j].imshow(feature_map,cm.gray, interpolation='quadric')
            axis[i, j].axis('off')    
    feature_maps.savefig(os.getcwd() + '/results/Part 11 - feature_maps.png')       
    plt.show() 
    
    return None


# In[42]:

plot_feature_map_at_different_locations(conv4_feature_maps)


