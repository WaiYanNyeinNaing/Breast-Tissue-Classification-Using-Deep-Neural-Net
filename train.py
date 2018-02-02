"""
Deep Learning with tensorflow (Intro - 2)
waiyannyeinnaing

Breast Tissue Classification and Breast Cancer Detection Using Multilayer Perceptron Deep Neural Network

Machien Learning Repository Link
http://archive.ics.uci.edu/ml/datasets/breast+tissue
"""

#Import Library
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#one_hot_encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels]=1
    return one_hot_encode
    

#Read the dataset
def read_dataset():
    data = pd.read_csv("C:/Users/ASUS.WINCTRL-LAAA72L/Downloads/Machine Learning/TensorFlow Step by Step tuto/breast_data.csv")
    #print(len(data.columns)) #number of features
    x = data[data.columns[0:9]].values
    y = data[data.columns[[9]]]
    #Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = one_hot_encode(y)
    return (x,y)


#Read Input/Target fro dataset
X,Y = read_dataset()

#Shuffle dataset
X,Y = shuffle(X,Y,random_state=1)       

"""
print(X.shape) #(106 samples,9 features)   // X.shape[0,1]
print(Y.shape) #(106 samples,6 labels)     // Y.shape[0,1]
"""

#Convert dataset into train and test part (20% data as test set)
train_x , test_x , train_y , test_y = train_test_split(X,Y,test_size=0.20,random_state=415)

"""
#Check the shape of train and test data
print("Shape of train_x is:  ")
print(train_x.shape)                        # (84 samples , 9 features)  
print("Shape of train_y is:  ")
print(train_y.shape)                        # (84 samples , 6 labels)    
print("Shape of test_x is:  ") 
print(test_x.shape)                         # (22 samples , 9 features)                                            
print("Shape of test_y is:  ")
print(test_y.shape)                         # (22 samples , 6 labels)
"""

#Define the hyper parameters to work with tensors
learning_rate = 0.01               #0.01
epochs = 1000                              
model_path = "C:\\Users\\ASUS.WINCTRL-LAAA72L\\Desktop\\Cancer predict\\model"     #to save pretrain model

#Input/Target Variables
n_features = X.shape[1]                                      #9 features
n_class = Y.shape[1]                                         #6 labels  
print("n_features is %s" % (n_features))
print("n_class is %s" % (n_class))


#Define the number of hidden layers and number of neurons of each layer
n_h_1 = 1500      #hidden layer 1    #1500
n_h_2 = 1000      #hidden layer 2    #1000
n_h_3 = 500      #hidden layer 3     #500
n_h_4 = 100      #hidden layer 4     #100 


#Intialize the parameter

# Shape of x , y_ placeholders
x = tf.placeholder(tf.float32,[None,n_features])         # (data.type, placeholder(number of samples,n_features))    #x = input features
y = tf.placeholder(tf.float32,[None,n_class])            # (data.type, placeholder(nunber of samples,n_class))       #y = actual class


#Shape of Weight & bias variables
W = tf.Variable(tf.zeros([n_features,n_class]))
b = tf.Variable(tf.zeros([n_class]))


#Define Weights and biases of each layer
Weights = {
    'h1': tf.Variable(tf.truncated_normal([n_features,n_h_1])),
    'h2': tf.Variable(tf.truncated_normal([n_h_1,n_h_2])),
    'h3': tf.Variable(tf.truncated_normal([n_h_2,n_h_3])),
    'h4': tf.Variable(tf.truncated_normal([n_h_3,n_h_4])),
    'out': tf.Variable(tf.truncated_normal([n_h_4,n_class]))
    }

biases = {
    'b1' : tf.Variable(tf.truncated_normal([n_h_1])),
    'b2' : tf.Variable(tf.truncated_normal([n_h_2])),
    'b3' : tf.Variable(tf.truncated_normal([n_h_3])),
    'b4' : tf.Variable(tf.truncated_normal([n_h_4])),
    'out' : tf.Variable(tf.truncated_normal([n_class]))
    }


#define the model
def multilayer_perceptron(x,Weights,biases):

    #Layer 1
    layer_1 = tf.add(tf.matmul(x, Weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)                                            #activation function

    #Layer 2
    layer_2 = tf.add(tf.matmul(layer_1, Weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    #Layer 3
    layer_3 = tf.add(tf.matmul(layer_2, Weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    #Layer 4
    layer_4 = tf.add(tf.matmul(layer_3, Weights['h4']),biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)

    #Layer 5 (final)
    final_layer = tf.add(tf.matmul(layer_4, Weights['out']),biases['out'])
    final_layer = tf.nn.softmax(final_layer)

    return final_layer


#predict model
predict = multilayer_perceptron(x,Weights,biases)
           

#Calculate the cost function and optimizer
cost_function = tf.nn.sigmoid_cross_entropy_with_logits(labels= y, logits=predict)               #logits = predict output
loss = tf.reduce_sum(cost_function)                                                           

#Calculate the optimizer (Using Gradient Descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)        
train = optimizer.minimize(loss)                                   #Training_step

#Intializer
init = tf.global_variables_initializer()  #important (need to initialize of all input data)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

#Calculate the cost error and accuracy for each epoch
mse_history = []
accuracy_history = []

for i in range(epochs):
    #Train the network
    sess.run(train,feed_dict={x:train_x,y:train_y})

    #calculate accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    accuracy_train = (sess.run(accuracy,feed_dict={x:train_x,y:train_y}))              #calulate the accuracy with train set
    accuracy_test = (sess.run(accuracy,feed_dict={x:test_x,y:test_y}))                 #calulate the accuracy with test set

    #Append to MSE history to plot
    predict_y = sess.run(predict,feed_dict={x:train_x})                                #Predict output Using Test Set
    mse = tf.reduce_mean(tf.square(predict_y - train_y))                                #calculate mean square error for prediction
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    #Append to accuracy hsitory to plot
    accuracy_history.append(accuracy_train)

    print("epoch: ",i," - " ,"Mean Square Error!: ",mse_,"Train Accuracy: ",accuracy_train,"Test Accuracy: ",accuracy_test)



#Save the pretrain mdoel (updated weights & biases)
save_path = saver.save(sess,model_path)                  #model path is the path// . . // to save the updated weights and biases
print("Model save in the file %s " % (save_path))


#Plot mse and acccuray graph
plt.plot(mse_history,"r")
plt.xlabel('number of iterations')
plt.ylabel('mean square error')
plt.show()
plt.plot(accuracy_history)
plt.xlabel('number of iterations')
plt.ylabel('correct detection accuracy')
plt.show()

#Print final Accuracy
correct_prediction = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("Final accuracy is ",(sess.run(accuracy,feed_dict={x:train_x,y:train_y})))                   #calulate the accuracy with test set

#Print final prediction and Mean Square error of test set
predict_y = sess.run(predict,feed_dict={x:test_x})
print("Prediction result of test set is: ", predict_y)
mse = tf.reduce_mean(tf.square(predict_y - test_y))                            #calculate mean square error for prediction
print("Final means square error is: ", (sess.run(mse)))


    
    
   
    
    










    

      

             
             
