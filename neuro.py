import numpy as np
import scipy.special as sp
import math 
import matplotlib.pyplot as plt


class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate,weight="simple",activate_function = "sigmoid"):
        self.innode = inputnodes
        self.hidnode = hiddennodes
        self.outnode = outputnodes
        #Below defines learning rate
        self.alpha = learningrate
        # The derivative and gredient update part need modification
        if activate_function == "sigmoid":
            self.activate_function = lambda x: sp.expit(x) #Use lambda expresssion to make a function replacable
        elif activate_function == "softmax":
            self.activate_function = lambda x: self.softmax(x)
        elif activate_function == 'tanh':
            self.activate_function = lambda x: np.tanh(x) # Use tanh function for updating
        #Below define the weight between different layers
        if weight  == 'simple':
            self.w1 = np.random.rand(self.hidnode,self.innode) - 0.5
            self.w2 = np.random.rand(self.outnode,self.hidnode) - 0.5
        else: #Better for training the example
            self.w1 = np.random.normal(0.0,power(self.hidnode,-0.5),(self.hidnode,self.innode))
            self.w2 = np.random.normal(0.0,power(self.outnode,-0.5),(self.outnode,self.hidnode))
        pass

    def train(self,input_list,target_list): #Targets are reference result
        targets = np.array(target_list,ndmin = 2).T #Need the transpose version
        inputs = np.array(input_list,ndmin = 2).T  #Need the Transpoes version
        #print(np.shape(targets),np.shape(inputs)) # Print out the size 
        hidden_output , final_output = self.query(input_list) #This output can be considered as a multi-classifier
        output_error = targets - final_output
        # Below Applied Back Propergation Algorithm
        hidden_error = np.dot(self.w2.T,output_error) # This is error back propergated to each node of hidden layer
        #print(np.shape(output_error),np.shape(hidden_error))
        # No need for input layer since input layer are different inputs.
        # The learning rate is a fixed number We can apply self adaptation learning
        # Below used the idea of gradient descent
        self.w2 += self.alpha * np.dot((output_error * final_output * (1.0 - final_output)),np.transpose(hidden_output))
        self.w1 += self.alpha * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)),np.transpose(inputs))
        pass


    def query(self,input_list): #This function can pass the value Throughout the neural Network
        inputs = np.array(input_list,ndmin = 2).T #Need The transpose version
        hidden_input = np.dot(self.w1,inputs)
        hidden_output = self.activate_function(hidden_input) #The input and output are all np matrix
        final_input = np.dot(self.w2,hidden_output)
        final_output = self.activate_function(final_input)
        return hidden_output , final_output #The result is returned as reference.

    # Below are math function defination
    # These provide more option for activate function
    def softmax(x,trans = False): # Be aware of the transpose or not
        if trans == True:
            return np.exp(x) / np.sum(np.exp(x), axis = 0)
        else:
            return np.exp(x).T / np.sum(np.exp(x) , axis = 0)

#Define and construct the neural network
input_nodes = 784
hidden_nodes = 100 # The value can be adjusted according the trainning result
output_nodes = 10  # If the situation changed to some pattern recognition the value can be modified.

#Define the learning rate for the model:
learning_rate = 0.3

#Create instance of Neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#Load the data from Mini MNIST
train_data_file = open('mnist_train.csv','r') # Use the train data set to train the network
test_data_file = open('mnist_mini.csv','r') #Use the mini data as examine the model.
train_data_list = train_data_file.readlines() #Read the data to be line set
test_data_list = test_data_file.readlines()   #Same the thing
train_data_file.close() #Close the file to clean more memory
test_data_file.close()  

#TRAIN THE NETWORK !!! KEY
for record in train_data_list:
    raw_value = record.split(',')
    #The scaled data set for better neural network performance
    inputs = (np.asfarray(raw_value[1:])) / 255.0 * 0.99 + 0.01
    #Get the reference from first bit.
    targets = np.zeros(output_nodes) + 0.01
    # Give set the reference for the successful item to be large probability
    targets[int(raw_value[0])] = 0.99 
    #print(np.shape(inputs),np.shape(targets))
    n.train(inputs.tolist(),targets.tolist())
    pass

# The drawback in this training condition : No biased value item so that no constant item can be used to accelerate the 
# training process

# below is a visualize examine process
# It will get a data from the mini_mnist and display as a picture
# After the picture is closed it will run the query in Neural Network and check
# If the model yield an right answer

for record in test_data_list:
    raw_value = record.split(',')
    img_array = np.asfarray(raw_value[1:]).reshape((28,28))
    # Config the image to be an array readable by matplotlib
    plt.imshow(img_array,cmap = 'Greys',interpolation = 'None')
    #Display the picture until the picture is closed
    plt.show()
    hidden, result = n.query((np.asfarray(raw_value[1:])/ 255.0 * 0.99 + 0.01).tolist()) # get result from neural network
    rst_list = result.tolist()
    max_index = rst_list.index(max(rst_list))
    print("The result is :",max_index) # Print out the number of the max item

    

