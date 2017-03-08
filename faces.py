
# coding: utf-8

# # Face Recognitions with Tensorflow

# ## Import Pakages

## Note: The following code is written in python 2 and with tensorflow 1.0 in
## Anaconda distributions
 
## File "faces_subset.txt" need to be under the same directory as this file


# In[1]:

import hashlib
import urllib
from numpy import * 
import numpy as np
import pandas as pd
from pylab import *
import csv
import glob
from scipy.ndimage import filters
from scipy.misc import imread
from scipy.misc import imresize,imsave
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random
import cPickle
act = list(set([a.split("\t")[0] for a in open("faces_subset.txt").readlines()]))


# ## Download Images and Save Under Uncropped Folder

# In[139]:
    
## Part 7

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def download_uncropped_images(input_file):
    testfile = urllib.URLopener()            
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work
    crop_coordinates = {}
    existed_url = {}
    actor_name = {}
    hash_value = {}
    
    if not os.path.exists(os.getcwd() + '/uncropped'):
        os.makedirs(os.getcwd() + '/uncropped')
    else:
        pass
    
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(input_file):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                if not line.split()[4] in existed_url.values():
                    timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 45)
                    crop_coordinates[filename] = line.split()[5]
                    existed_url[filename] = line.split()[4]
                    actor_name[filename] = line.split()[0] + ' ' + line.split()[1]
                    hash_value[filename]= line.split()[6]
                  
                    
                    if not os.path.isfile("uncropped/"+filename):
                        continue
                else: 
                    continue              
                    
                print (filename + " saved under uncropped")
                i += 1 
                                
    if not os.path.exists(os.getcwd() + '/Namelist'):
         os.makedirs(os.getcwd() + '/Namelist')
    else: 
        pass
   

    crop_coordinates_dataframe = pd.DataFrame.from_dict(crop_coordinates,orient='index')
    crop_coordinates_dataframe.columns = ['crop_coordinates']
    crop_coordinates_dataframe['image_name'] = crop_coordinates_dataframe.index
    crop_coordinates_dataframe.to_csv(os.getcwd() +'/Namelist/crop coordinates.csv', index = False) 
                         
    actor_name_dataframe =  pd.DataFrame.from_dict(actor_name,orient='index')
    actor_name_dataframe.columns = ['actor_names']
    actor_name_dataframe['image_name'] = actor_name_dataframe.index
    actor_name_dataframe.to_csv(os.getcwd() +'/Namelist/actor name.csv', index = False) 
    
    hash_value_dataframe =  pd.DataFrame.from_dict(hash_value,orient='index')
    hash_value_dataframe.columns = ['hash_value']
    hash_value_dataframe['image_name'] = hash_value_dataframe.index
    hash_value_dataframe.to_csv(os.getcwd() +'/Namelist/hash value.csv', index = False) 
    
    return None


# In[140]:

download_uncropped_images(os.getcwd() + "/faces_subset.txt")


# ### Remove Bad Images

# In[141]:

def get_all_images(paths):
    image_list = glob.glob(paths)
    return image_list

def remove_bad_images(image_list, hash_values):
    for i in image_list:
        
        image = open(i).read()
        image_name = i.split('/')[-1]
        exp_hash = hash_values[image_name]
        m = hashlib.sha256()
        m.update(image)
        
        if  m.hexdigest() != exp_hash:
            print(image_name + " is not the right picture, removed")
            os.remove("uncropped/"+image_name)

    return None


# ## Convert Uncropped Images to Cropped Images, and Apply Gray Scale

# ### Helper Functions

# In[142]:

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def crop_and_gray_image(image_path,crop_coordinates, im_size_h, im_size_v):  
    image = imread(image_path)
    image_name = image_path.split('/')[-1]
    x1 = int(crop_coordinates[image_name].split(',')[0])
    y1 = int(crop_coordinates[image_name].split(',')[1])
    x2 = int(crop_coordinates[image_name].split(',')[2])
    y2 = int(crop_coordinates[image_name].split(',')[3])
    
    cropped  = image[y1:y2, x1:x2]
    std_cropped = imresize(cropped, (im_size_h,im_size_v))
    std_gray_cropped = rgb2gray(std_cropped)
    imshow(std_gray_cropped, cm.gray)

    return [image_name, std_gray_cropped]


# ### Update All Images 

# In[143]:

def update_images(image_list, crop_coordinates,  im_size_h, im_size_v):
    if not os.path.exists(os.getcwd() + '/cropped'):
        os.makedirs(os.getcwd() + '/cropped')
    else:
        pass
    crop_path = os.getcwd() + '/cropped/'

    for i in image_list:
        try:
                plt.show()
                image_name, std_gray_cropped = crop_and_gray_image(i,crop_coordinates, im_size_h, im_size_v)
                imsave(crop_path + image_name, std_gray_cropped)
        except:
            pass     
        
    print ("image update completed")
    
    return None


# ### Execute update images function 

# In[144]:

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
remove_bad_images(image_list_uncropped, hash_values)
im_size_h = 64
im_size_v = 64 
image_list_updated = get_all_images(os.getcwd() + "/uncropped/*")
update_images(image_list_updated,crop_coordinates, im_size_h, im_size_v)


# In[145]:

def clear_non_readable_files(actor_name_dataframe,image_list):
    image_name_list = []
    for i in image_list:
        image_name = i.split('/')[-1]
        image_name_list.append(image_name)
    actor_dataframe = actor_name_dataframe[actor_name_dataframe['image_name'].isin(image_name_list)]    
    return [image_name_list, actor_dataframe]


# ## Build training, validation and test sets

# ### build actor name dataframes

# In[146]:

uncropped_path = os.getcwd() + "/uncropped/*"
cropped_path = os.getcwd() + "/cropped/*"
act = list(set([a.split("\t")[0] for a in open(os.getcwd() + "/faces_subset.txt").readlines()]))
image_list = get_all_images(uncropped_path)


# In[147]:

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

# In[148]:

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


# ###  Build training, validation and test sets for all actors in forms of dataframes

# In[149]:

def organize_all_data(actor_list, actor_name_dataframe, image_name_list, cropped_path):
    organized_data = {}
    for i in actor_list:
        organized_data[i] = separate_training_sets(i,actor_name_dataframe, image_name_list, cropped_path)
        
    return organized_data


# In[150]:

def one_hot_encode_set(df):
    encoded_cols =  pd.get_dummies(df['actor_names']) 
    one_hot_full_data = df.join(encoded_cols)    
    return one_hot_full_data


# In[151]:

organized_data = organize_all_data(act, actor_dataframe, image_name_list, cropped_path)


# In[152]:

act6 =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
training_df = [organized_data[i][0] for i in act6]
test_df = [organized_data[i][1] for i in act6]
validation_df = [organized_data[i][2] for i in act6]
training_df = one_hot_encode_set(pd.concat(training_df))
test_df = one_hot_encode_set(pd.concat(test_df))
validation_df = one_hot_encode_set(pd.concat(validation_df))


# In[153]:

def reshape_image(image_path, im_size_h, im_size_v):
    image = imread(image_path)/255.0
    reshpaed_image = np.reshape(image,(1,im_size_h*im_size_v))
    return reshpaed_image


# In[154]:

def get_training_set_parameters(training_df, im_size_h, im_size_v, seed): 
    random.seed(seed)
    grouped = training_df.groupby('actor_names') 
    training_sampled = pd.concat([d.ix[random.sample(d.index, 4)] for _, d in grouped]).reset_index(drop=True)
    
    train_x = np.ones((1, im_size_h*im_size_v))
    for j in training_sampled['image_path']:
        reshaped_image = reshape_image(j, im_size_h, im_size_v)
        train_x = np.vstack((train_x, reshaped_image))
    train_x= train_x[1:,:]
    ycols = training_sampled.columns.values.tolist()[-6:]
    train_y = training_sampled[ycols].as_matrix()
    
    return [train_x, train_y]


# In[155]:

def get_test_set_parameters(test_df, im_size_h, im_size_v): 
    test_x = np.ones((1, im_size_h*im_size_v))
    for j in test_df['image_path']:
        reshaped_image = reshape_image(j, im_size_h, im_size_v)
        test_x = np.vstack((test_x, reshaped_image))
    test_x= test_x[1:,:]
    ycols = test_df.columns.values.tolist()[-6:]
    test_y = test_df[ycols].as_matrix()
    
    return [test_x, test_y]


# ## Neural Network Training

# ### Initialize x, y, W and b

# In[156]:

x = tf.placeholder(tf.float32, [None, im_size_h*im_size_v])
y_predict = tf.placeholder(tf.float32, [None, 6])


# In[157]:

num_hidden_units = 100

# weights for the edges from input to the first hidden layer, initalized by random normal
W0 = tf.Variable(tf.random_normal([im_size_h*im_size_v, num_hidden_units], stddev=0.00001, seed = 1)/1000000)
b0 = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.00001, seed = 1)/10000000)

# weights for the edges from the first hidden layer to output
W1 = tf.Variable(tf.random_normal([num_hidden_units, 6], stddev=0.00001, seed = 1)/1000000)
b1 = tf.Variable(tf.random_normal([6], stddev=0.00001, seed = 1)/1000000)


# ### Define layers

# In[158]:

layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1
y_true = tf.nn.softmax(layer2)


# ### Cost functions and train the nerual network

# In[159]:

lamda = 0.01
large_w_penalty =lamda*tf.reduce_sum(tf.square(W0))+lamda*tf.reduce_sum(tf.square(W1))
log_likelihood_cost = -tf.reduce_sum(y_predict*tf.log(y_true))+large_w_penalty
train_step = tf.train.AdamOptimizer(0.001).minimize(log_likelihood_cost)


# In[160]:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test_set_parameters(test_df,  im_size_h, im_size_v)
validation_x, validation_y = get_test_set_parameters(validation_df,  im_size_h, im_size_v)
full_train_x, full_train_y = get_test_set_parameters(training_df,  im_size_h, im_size_v)


# In[161]:

def train_neural_network(test_x, test_y, validation_x, validation_y, full_train_x, full_train_y):
    train_learning_curve = {}
    test_learning_curve = {}
    validation_learning_curve = {}
    for i in range(1001):
        train_x, train_y = get_training_set_parameters(training_df, im_size_h, im_size_v, i)
        sess.run(train_step, feed_dict={x: train_x, y_predict: train_y})

        if i % 50 == 0 :
                print "i=",i

                print "Train:", str(sess.run(accuracy, feed_dict={x: full_train_x, y_predict: full_train_y})*100)+"%"
                train_learning_curve [i] = sess.run(accuracy, feed_dict={x: full_train_x, y_predict: full_train_y})*100
                print "Test:", str(sess.run(accuracy, feed_dict={x: test_x, y_predict: test_y}) *100) + "%"
                test_learning_curve[i] = sess.run(accuracy, feed_dict={x: test_x, y_predict: test_y})*100
                print "Validation:", str(sess.run(accuracy, feed_dict={x: validation_x, y_predict: validation_y}) *100) + "%"
                validation_learning_curve[i] = sess.run(accuracy, feed_dict={x: validation_x, y_predict: validation_y})*100
                print "Penalty:", sess.run(large_w_penalty)
                print "\n"
                snapshot = {}
                snapshot["W0"] = sess.run(W0)
                snapshot["W1"] = sess.run(W1)
                snapshot["b0"] = sess.run(b0)
                snapshot["b1"] = sess.run(b1)
                if not os.path.exists(os.getcwd() + '/weights'):
                        os.makedirs(os.getcwd() + '/weights')
                else:
                         pass
                cPickle.dump(snapshot,  open("weights/tensorflow_weight"+str(i)+".pkl", "w"))
    learning_curves = [train_learning_curve, test_learning_curve, validation_learning_curve]           
    return [snapshot, learning_curves]


# In[162]:

snapshot, learning_curves  =  train_neural_network(test_x, test_y, validation_x, validation_y, full_train_x, full_train_y)


# In[163]:

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
    plt.legend(loc = 'best')
    figure = plt.gcf()
    plt.show()
    figure.savefig(os.getcwd() + '/results/Part 7 - Neural Network Learning Curves.png')
    
    return None


# In[164]:

plot_learning_curves(learning_curves)


# # Find the Neurons that are useful to a certain actor

# ## Assume the actor is America Ferrera

# In[165]:
    
## Part 9

def find_neuron_for_Ferrera(W0):
    W0 = snapshot['W0']
    neruons = {}
    Ferrera_y = array([0, 1, 0, 0, 0, 0]).reshape(1,6)
    for i in range(W0.shape[1]):
        neruons[i] = sess.run(log_likelihood_cost, feed_dict={x: W0[:,i].reshape(1, W0.shape[0]), y_predict: Ferrera_y})
    most_like_Ferrera = min(neruons, key=neruons.get)
    imshow(W0[:, most_like_Ferrera].reshape(64,64), cm.coolwarm, interpolation='quadric')
    figure = plt.gcf()
    plt.show()
    figure.savefig(os.getcwd() + '/results/Part 9 - Weights for Ferrera.png')

    
    return None


# In[166]:

W0 = snapshot['W0']
find_neuron_for_Ferrera(W0)


# ## Assume the actor is Alec Baldwin

# In[167]:

def find_neuron_for_Baldwin(W0):
    W0 = snapshot['W0']
    neruons = {}
    Ferrera_y = array([1, 0, 0, 0, 0, 0]).reshape(1,6)
    for i in range(W0.shape[1]):
        neruons[i] = sess.run(log_likelihood_cost, feed_dict={x: W0[:,i].reshape(1, W0.shape[0]), y_predict: Ferrera_y})
    most_like_Baldwin = min(neruons, key=neruons.get)
    imshow(W0[:, most_like_Baldwin].reshape(64,64), cm.coolwarm, interpolation='quadric')
    figure = plt.gcf()
    plt.show()
    figure.savefig(os.getcwd() + '/results/Part 9 - Weights for Baldwin.png')
    
    return None


# In[168]:

W0 = snapshot['W0']
find_neuron_for_Baldwin(W0)


# # Test Cases where regularizations are important

# In[185]:

## Part 8
    
    
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, im_size_h*im_size_v])
y_predict = tf.placeholder(tf.float32, [None, 6])
num_hidden_units = 100

# weights for the edges from input to the first hidden layer, initalized by random normal
W0 = tf.Variable(tf.random_normal([im_size_h*im_size_v, num_hidden_units], stddev=0.00001, seed = 1)/1000000)
b0 = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.00001, seed = 1)/10000000)

# weights for the edges from the first hidden layer to output
W1 = tf.Variable(tf.random_normal([num_hidden_units, 6], stddev=0.00001, seed = 1)/1000000)
b1 = tf.Variable(tf.random_normal([6], stddev=0.00001, seed = 1)/1000000)

layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1
y_true = tf.nn.softmax(layer2)


# In[186]:

training_df_sample = training_df.sample(n = 120, random_state = 23)
test_df_sample = test_df.sample(n = 30, random_state = 23)

# In[187]:

training_df_sample.shape


# In[188]:

num_hidden_units = 100


# In[189]:

def get_training_set_parameters_part8(training_df, im_size_h, im_size_v, seed): 
    random.seed(seed)
    grouped = training_df.groupby('actor_names') 
    training_sampled = pd.concat([d.ix[random.sample(d.index, 1)] for _, d in grouped]).reset_index(drop=True)
    
    train_x = np.ones((1, im_size_h*im_size_v))
    for j in training_sampled['image_path']:
        reshaped_image = reshape_image(j, im_size_h, im_size_v)
        train_x = np.vstack((train_x, reshaped_image))
    train_x= train_x[1:,:]
    ycols = training_sampled.columns.values.tolist()[-6:]
    train_y = training_sampled[ycols].as_matrix()
    
    return [train_x, train_y]


# In[190]:

def train_neural_network_part8(test_x, test_y, validation_x, validation_y, full_train_x, full_train_y):
    train_learning_curve = {}
    test_learning_curve = {}
    validation_learning_curve = {}
    for i in range(1501):
        train_x, train_y = get_training_set_parameters(training_df, im_size_h, im_size_v, i)
        sess.run(train_step, feed_dict={x: full_train_x, y_predict: full_train_y})

        if i % 50 == 0 :
                print "i=",i

                print "Train:", str(sess.run(accuracy, feed_dict={x: full_train_x, y_predict: full_train_y})*100)+"%"
                train_learning_curve [i] = sess.run(accuracy, feed_dict={x: full_train_x, y_predict: full_train_y})*100
                print "Test:", str(sess.run(accuracy, feed_dict={x: test_x, y_predict: test_y}) *100) + "%"
                test_learning_curve[i] = sess.run(accuracy, feed_dict={x: test_x, y_predict: test_y})*100
                print "Penalty:", sess.run(large_w_penalty)
                print "\n"
                snapshot = {}
                snapshot["W0"] = sess.run(W0)
                snapshot["W1"] = sess.run(W1)
                snapshot["b0"] = sess.run(b0)
                snapshot["b1"] = sess.run(b1)
                if not os.path.exists(os.getcwd() + '/weights'):
                        os.makedirs(os.getcwd() + '/weights')
                else:
                         pass
                cPickle.dump(snapshot,  open("weights/tensorflow_weight"+str(i)+".pkl", "w"))
    learning_curves = [test_learning_curve]           
    return [snapshot, learning_curves]


# ### No lambda

# In[191]:

lamda = 0.00
large_w_penalty =lamda*tf.reduce_sum(tf.square(W0))+lamda*tf.reduce_sum(tf.square(W1))
log_likelihood_cost = -tf.reduce_sum(y_predict*tf.log(y_true)) + large_w_penalty
train_step = tf.train.AdamOptimizer(0.0001).minimize(log_likelihood_cost)


# In[192]:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test_set_parameters(test_df_sample,  im_size_h, im_size_v)
validation_x, validation_y = get_test_set_parameters(validation_df,  im_size_h, im_size_v)
full_train_x, full_train_y = get_test_set_parameters(training_df_sample,  im_size_h, im_size_v)


# In[193]:

full_train_y.shape


# In[194]:

snapshot, learning_curves_no_reg  =  train_neural_network_part8(test_x, test_y, validation_x, validation_y, full_train_x, full_train_y)


# ### With lambda

# In[178]:

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, im_size_h*im_size_v])
y_predict = tf.placeholder(tf.float32, [None, 6])
num_hidden_units = 100

# weights for the edges from input to the first hidden layer, initalized by random normal
W0 = tf.Variable(tf.random_normal([im_size_h*im_size_v, num_hidden_units], stddev=0.00001, seed = 1)/1000000)
b0 = tf.Variable(tf.random_normal([num_hidden_units], stddev=0.00001, seed = 1)/10000000)

# weights for the edges from the first hidden layer to output
W1 = tf.Variable(tf.random_normal([num_hidden_units, 6], stddev=0.00001, seed = 1)/1000000)
b1 = tf.Variable(tf.random_normal([6], stddev=0.00001, seed = 1)/1000000)

layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1
y_true = tf.nn.softmax(layer2)


# In[179]:

lamda = 0.25
large_w_penalty = lamda*tf.reduce_sum(tf.square(W0)) + lamda*tf.reduce_sum(tf.square(W1))
log_likelihood_cost = -tf.reduce_sum(y_predict*tf.log(y_true)) + large_w_penalty
train_step = tf.train.AdamOptimizer(0.0001).minimize(log_likelihood_cost)


# In[180]:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test_set_parameters(test_df_sample,  im_size_h, im_size_v)
validation_x, validation_y = get_test_set_parameters(validation_df,  im_size_h, im_size_v)
full_train_x, full_train_y = get_test_set_parameters(training_df_sample,  im_size_h, im_size_v)


# In[181]:

snapshot, learning_curves_reg  =  train_neural_network_part8(test_x, test_y, validation_x, validation_y, full_train_x, full_train_y)


# In[225]:

def compare_learning_curves(learning_curves_reg, learning_curves_no_reg):
    
    if not os.path.exists(os.getcwd() + '/results'):
         os.makedirs(os.getcwd() + '/results')
    else: 
        pass 
    no_reg_test_performances = learning_curves_no_reg[0]
    no_reg_test_list = sorted(no_reg_test_performances.items())
    reg_test_performances = learning_curves_reg[0]
    reg_test_list = sorted(reg_test_performances.items())
    n, accuracy_no_reg = zip(*no_reg_test_list)
    n, accuracy_reg = zip(*reg_test_list)
    plt.figure(figsize=(10,6))
    plt.title('Accuracy of Test Set With and Without Regularizations')
    plt.xlabel('Number of Iterations n')
    plt.ylabel('Accuracy %')
    
    plt.plot(n, accuracy_reg, label = 'With Regularizations')
    plt.plot(n, accuracy_no_reg, label = 'Without Regularizations')
    plt.legend(loc = 'best')
    figure = plt.gcf()
    
    figure.savefig(os.getcwd() + '/results/Part 8 - Compare Learning Curves.png')
    
    return None


# In[226]:

compare_learning_curves(learning_curves_reg, learning_curves_no_reg)


# In[ ]:



