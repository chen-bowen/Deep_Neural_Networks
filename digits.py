
# coding: utf-8
## File "mnist_all.mat" need to be under the same directory as this file

# In[2]:


"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
from numpy import random
import os
import pandas as pd
from scipy.linalg import norm
import scipy.stats

# Part 1

def add_labels(dataset):
    # Given the dataset input as a dictionary, add the true label to the 
    # corrresponding data and output them as dataframes
    df_list = []
    for i in range(10): 
        label = np.ones((1, np.shape(dataset[str(i)])[0]))*i
        single_df = pd.DataFrame({'number_im_data':dataset[str(i)].tolist(), 'label':label.tolist()[0]})
        df_list.append(single_df)
        
    set_df = pd.concat(df_list)
    set_df = set_df.reset_index(drop=True)
    
    return set_df


def build_training_test_sets():
    # Download image data and separate them as training set and test set, in the
    # format of dataframes
	mat = scipy.io.loadmat('mnist_all.mat')
	mat.pop('__globals__')
	mat.pop('__header__')
	mat.pop('__version__')
	
	training_set = {}
	test_set = {}
	
	for key in mat:
	    if key[:5] == 'train':
	       training_set[key[-1]] = mat[key]/255.0
	    else:
	        test_set[key[-1]] = mat[key]/255.0
	
	training_df = add_labels(training_set)
	test_df = add_labels(test_set)
	
	return [training_df, test_df]

def plot_10_per_digits(training_df):
    
    if not os.path.exists(os.getcwd() + '/results'):
         os.makedirs(os.getcwd() + '/results')
    else: 
        pass 

    sample_image, axis = plt.subplots(10, 10, figsize=(15,15))
    
    for i in range(10):
        sample_set = training_df[training_df['label'] == i]
        sub_sample = sample_set.sample(n = 10, random_state = 1)
        images = sub_sample['number_im_data'].tolist()
        images = np.array(images, dtype = 'float64')
        for j in range(10):
           axis[i, j].imshow(images[j,:].reshape(28,28), cmap = cm.Greys_r)
           axis[i, j].axis('off')
    
    sample_image.savefig(os.getcwd() + '/results/Part 1 - Sample Digits.png')
    
    return None

def get_regression_parameters(dataset):
    # build regression parameters x and y with the input dataframes
    
    encoded_cols = pd.get_dummies(dataset['label'].astype(int))
    dataset = dataset.join(encoded_cols)
    
    x = dataset['number_im_data'].tolist()
    x = np.array(x, dtype = 'float64')
    all_one = np.ones(np.shape(x)[0])
    x = np.vstack((all_one,x.T)).T
    
    ycols = dataset.columns.values.tolist()[-10:]
    y = dataset[ycols].as_matrix()

    return [x, y]

#Part 2

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm


def f(x,y,w):
    m = np.shape(x)[1]
    z = np.dot(x,w)
    p = softmax(z)
    
    return -1.0/m *np.sum(y*np.log(p)) 

#Part3    
def df(x,y,w):
    m = np.shape(x)[1]
    z = np.dot(x,w)
    p = softmax(z)
    
    return -1.0/m * np.dot(x.T,(y-p))    



def finite_difference(x, y, w):
    delta_h = 0.0000001
    df_FD = np.zeros(np.shape(w))
    for i in range(np.shape(w)[0]):
        for j in range(np.shape(w)[1]):          
            h = np.zeros(np.shape(w))
            h[i][j] = delta_h
            df_FD[i][j] = (f(x, y, w+h) - f(x,y, w-h))/(2*delta_h)
    return df_FD

def gradient_descent(f, df, x, y, init_w, alpha, max_iter):
        EPS = 1e-5
        prev_w = init_w - 10 * EPS
        w = init_w.copy()  
        iter = 0
        while  norm(w-prev_w) > EPS and iter < max_iter : 
            prev_w = w.copy()
            w -= alpha*df(x, y, w)
            if iter % 500 == 0:
                print ("Iter:")
                print (iter)
                print ("Cost function value:")
                print (f(x,y,w))
                print ("Weights: ")
                print (w [0:4, 0])
                print ("Gradient: ")
                print (df(x, y, w) [0:4, 0])
                print ("\n")
            iter += 1
        print ("End Iteration", iter)
        print ("Cost Function Value: ", f(x, y, w))
        return w    
    


# Part 4

def train_model (train_x, train_y, f, df, alpha, max_iter):
    
    w_size = (785,10)
    initial_w = np.zeros((w_size[0], w_size[1]))/(np.shape(train_x)[1]**2)
    w = gradient_descent(f, df, train_x, train_y, initial_w, alpha, max_iter)
    
    return w


def train_model_record_performance(f, df, train_x, train_y,test_x,test_y, init_w, alpha, max_iter, save_iterations):
        EPS = 1e-5
        prev_w = init_w - 10 * EPS
        w = init_w.copy()  
        iteration = 0
        learning_curve = {}
        weights = []
        while  norm(w-prev_w) > EPS and iteration <= max_iter : 
            prev_w = w.copy()
            w -= alpha*df(train_x, train_y, w)
            w_save = w.copy()
            if iteration in save_iterations:
                
                performance_train = evaluate_model(train_x,train_y, w)
                performance_test = evaluate_model(test_x,test_y, w)
                weights.append(w_save)
                learning_curve [iteration] = [performance_train, performance_test]
                print ("Iter:")
                print (iteration)
                print ("Cost function value:")
                print (f(train_x, train_y, w))
                print ("The Training Set Performance is: " + str(performance_train*100) + "%" )
                print ("The Test Set Performance is: " + str(performance_test*100) + "%" )
                print ("\n")
                
            iteration += 1

        print ("End Iteration", iter)
        print ("Cost Function Value: ", f(train_x, train_y, w))
        return [weights, learning_curve]    


def get_learning_curve(training_df, test_df, train_regression_model, evaluate_performace, f, df):
    save_iterations = [1, 10, 20, 50, 100, 200, 500, 1000, 2500, 5000] 
    max_iter =  5000
    train_x, train_y = get_regression_parameters(training_df)
    test_x,test_y = get_regression_parameters(test_df)
    w_size = (785,10)
    init_w = np.zeros((w_size[0], w_size[1]))/(np.shape(train_x)[1]**2)
    alpha = 0.001
    weights, learning_curve = train_model_record_performance(f, df, train_x, train_y,test_x,test_y, init_w, alpha, max_iter, save_iterations)
    return [weights, learning_curve] 

def plot_learning_curve(learning_curve):
    
    if not os.path.exists(os.getcwd() + '/results'):
         os.makedirs(os.getcwd() + '/results')
    else: 
        pass 
    
    training_performances = {}
    validation_performances = {}
    for key in learning_curve:
        training_performances[key] = learning_curve[key][0]*100
        validation_performances[key] = learning_curve[key][1]*100
    training_list = sorted(training_performances.items())
    validation_list = sorted(validation_performances.items())
    n, accuracy_training = zip(*training_list)
    n, accuracy_validation = zip(*validation_list)
    plt.figure(figsize=(10,6))
    plt.title('Accuracy of Training Set and Test Set')
    plt.xlabel('Number of Iterations n')
    plt.ylabel('Accuracy %')
    
    plt.plot(n, accuracy_training, label = 'Training Set Performances')
    plt.plot(n, accuracy_validation, label = 'Validation Set Performances')
    plt.legend(loc = 'bottom right')
    figure = plt.gcf()
    plt.show()
    figure.savefig(os.getcwd() + '/results/Part 4 - Learning Curve.png')
    
    return None
 
def display_weights_iter(weights):
    
    shape = [28,28]

    weights_image, axis = plt.subplots(10, 10, figsize=(20,20))
    
    for i in range(10):   
        for j in range(10):
           w_graph = np.reshape(weights[j][1:,i], shape)
           axis[i, j].imshow(w_graph, cmap = cm.coolwarm)
           axis[i, j].axis('off')       
    weights_image.savefig(os.getcwd() + '/results/Part 4 - weights_image.png')       
           
        
    return None    

def evaluate_model(test_x,test_y,w):

    probability_test_y = softmax(np.dot(test_x,w))                     
    hypothesis_y_test = np.zeros_like(probability_test_y)
    hypothesis_y_test[np.arange(len(probability_test_y)), probability_test_y.argmax(1)] = 1 
    total_correct = 0 
    for i in range(len(hypothesis_y_test)):
        if np.array_equal(test_y[i], hypothesis_y_test[i]):
            total_correct += 1
    return total_correct*1.0/np.shape(test_y)[0]

def evaluate_model_trials(simu_test_x, simu_test_y,w):

    test_y_adj = np.zeros_like(simu_test_y)
    test_y_adj[np.arange(len(simu_test_y)), simu_test_y.argmax(1)] = 1 
    probability_test_y = softmax(np.dot(simu_test_x,w))                     
    hypothesis_y_test = np.zeros_like(probability_test_y)
    hypothesis_y_test[np.arange(len(probability_test_y)), probability_test_y.argmax(1)] = 1 
    total_correct = 0 
    for i in range(len(hypothesis_y_test)):
        if np.array_equal(test_y_adj[i], hypothesis_y_test[i]):
            total_correct += 1
    return total_correct*1.0/np.shape(test_y_adj)[0]

#Part 5
def f_SSE(x,y,w):
    m = np.shape(x)[1]
    return 1.0/m * sum((y- np.dot(x,w)) ** 2)
    
def df_SSE(x,y,w):
    m = np.shape(x)[1]
    return -2.0/m *np.dot(x.T,(y- np.dot(x,w)))   

def simulation_trial_two_models():
    
    sigma = 0.1

    w_size = (785,10)
    part5_w = np.random.random(w_size)    
    train_x_size = (12000, 785)
    train_y_size = (12000, 10)
    test_x_size = (1000, 785)
    test_y_size = (1000, 10)
    
#   Normal
    simu_train_x = np.random.random(train_x_size)
    simu_train_y = softmax(np.dot(simu_train_x, part5_w) + scipy.stats.norm.rvs(scale = sigma, size=train_y_size))   
    simu_test_x = np.random.random(test_x_size)
    simu_test_y = softmax(np.dot(simu_test_x, part5_w) + scipy.stats.norm.rvs(scale = sigma, size=test_y_size))    

    alpha_softmax = 0.0005
    max_iter_soft = 20000
    w_softmax = train_model (simu_train_x,simu_train_y, f, df, alpha_softmax, max_iter_soft)
    alpha_linear = 0.00001
    max_iter_linear = 100000
    w_linear = train_model(simu_train_x,simu_train_y, f_SSE, df_SSE, alpha_linear, max_iter_linear)

    multi_performance_train = evaluate_model_trials (simu_train_x,simu_train_y,w_softmax) 
    multi_performance_test = evaluate_model_trials (simu_test_x,simu_test_y,w_softmax)
    linear_performance_train = evaluate_model_trials (simu_train_x,simu_train_y,w_linear)
    linear_performance_test = evaluate_model_trials (simu_test_x,simu_test_y,w_linear)
    
    print ('Training Set performance using multinomial method is:'+ str(multi_performance_train *100) + "% \n" )
    print ('Test Set performance using multinomial method is:'+ str(multi_performance_test *100) + "% \n")
    print ('Training Set performance using linear method is:'+ str(linear_performance_train *100) + "% \n")
    print ('Test Set performance using linear method is:'+ str(linear_performance_test *100) + "% \n")
    
    return [multi_performance_train, multi_performance_test, linear_performance_train,  linear_performance_test]
    
if __name__ == "__main__":
    
# Part 1 & Part 2    
    
    training_df, test_df = build_training_test_sets()
    plot_10_per_digits(training_df)
    train_x, train_y = get_regression_parameters(training_df)
    test_x, test_y = get_regression_parameters(test_df)
    


# Part 3
    random.seed(1)

    x = train_x[0:4, 0:5]
    y = train_y[0:4, 0:6]
    w = np.random.rand(5,6) 

    df_trial = df(x,y,w)
    df_FD_trial = finite_difference(x, y, w)
    
    print ('The derivative of cost funtion computated by df_multi_class is \n', df_trial)
    print ('The derivative of cost funtion computated by finite difference is \n',  df_FD_trial)
    error = df_trial - df_FD_trial
    print ('The difference between two methods is \n',error)   
    
  
# Part 4 

    weights, learning_curve = get_learning_curve(training_df, test_df, train_model, evaluate_model, f, df)
    plot_learning_curve(learning_curve)
    display_weights_iter(weights)
    
#    
# Part 5
    np.random.seed(0)
    multi_performance_train, multi_performance_test, linear_performance_train,  linear_performance_test = simulation_trial_two_models()

    display_weights_iter(weights)






