import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward, load_data, timer


class DeepNN():
    def __init__(self,X_orig = None, Y_orig = None, dims = None, parameters = None, learning_rate = None, num_iterations = None):

        self._X = self.__vector_normalize(X_orig)
        self._Y = Y_orig
        self._dims = dims
        if parameters == None:
            self._parameters = self.__initialize_parameters_deep()
        else:
            self._parameters = parameters
        self._learning_rate = learning_rate
        self._num_iterations = num_iterations

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X_val):
        if type(X_val) is np.ndarray:
            self._X = X_val
        else:
            print('X should be numpy array')

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Y_val):
        if type(Y_val) is np.ndarray:
            self._Y = Y_val
        else:
            print('Y should be numpy array')

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, d):
        if type(dims) is list:
            self._dims = d
        else:
            print('dims should be list')

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        if type(params) is dict:
            self._parameters = params
        else:
            print('parameters should be dict')

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, lr):
        if type(lr) is float:
            self._learning_rate = lr
        else:
            print('learning rate should be float')

    @property
    def iterations(self):
        return self._num_iterations

    @iterations.setter
    def iterations(self, iters):
        if type(iters) is int:
            self._num_iterations = iters
        else:
            print('iterations should be int')

    def __str__(self):
        return f'DeepNN with X: {self._X.shape}, Y: {self._Y.shape}\nLayers Info: {self._dims}, Learning rate: {self._learning_rate}, #Iterations: {self._num_iterations}'

    #Flatten and normalize the images
    def __vector_normalize(self, train_x_orig):
        # Reshape the training and test examples 
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions

        # Standardize data to have feature values between 0 and 1.
        train_x = train_x_flatten/255. 

        return train_x     

    # Initialize parameters for deep NN
    def __initialize_parameters_deep(self):
        """
        Arguments:
        self._dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (self._dims[l], self._dims[l-1])
                        bl -- bias vector of shape (self._dims[l], 1)
        """
        
        np.random.seed(1)
        parameters = {}
        L = len(self._dims)            # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self._dims[l], self._dims[l-1])/np.sqrt(self._dims[l-1]) #* 0.01
            parameters['b' + str(l)] = np.zeros((self._dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (self._dims[l], self._dims[l-1]))
            assert(parameters['b' + str(l)].shape == (self._dims[l], 1))

            
        return parameters

    # Linear forward
    def __linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
    
        Z = np.dot(W, A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache

    # Linear-activation forward
    def _linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    # L-layer model
    def L_model_forward(self):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = self._X

        L = len(self._parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            A, cache = self._linear_activation_forward(A_prev, self._parameters["W"+str(l)], self._parameters["b"+str(l)], activation = "relu")
            caches.append(cache)
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self._linear_activation_forward(A, self._parameters["W"+str(L)], self._parameters["b"+str(L)], activation = "sigmoid")
        caches.append(cache)

        assert(AL.shape == (1, self._X.shape[1]))

                
        return AL, caches

    # Cost function
    def compute_cost(self,AL):
        """
        Implement the cost function defined by equation (7).
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
        Returns:
        cost -- cross-entropy cost
        """

        Y = self._Y      
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[x]] into x).
        assert(cost.shape == ())
        
        return cost

    # Linear backward
    def __linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db

    # Linear-activation backward
    def _linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
            
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db

    # L-layer backward
    def L_model_backward(self, AL, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = self._Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = self._linear_activation_backward(dAL, caches[L-1], activation = 'sigmoid')
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = current_cache
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(grads["dA"+str(l+1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    # Update parameters
    def update_parameters(self, grads):
        """
        Update parameters using gradient descent
        
        Arguments:
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """
        
        L = len(self._parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self._parameters["W" + str(l+1)] = self._parameters["W" + str(l+1)] - self._learning_rate * grads["dW" + str(l+1)]
            self._parameters["b" + str(l+1)] = self._parameters["b" + str(l+1)] - self._learning_rate * grads["db" + str(l+1)]
        return self._parameters

    @timer
    def L_layer_model(self, print_cost=False):#lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []                         # keep track of cost
        
        
        # Loop (gradient descent)
        for i in range(0, self._num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward()

            # Compute cost.
            cost = self.compute_cost(AL)

            # Backward propagation.
            grads = self.L_model_backward(AL, caches)

            # Update parameters.
            self._parameters = self.update_parameters(grads)
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                
        #plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self._learning_rate))
        plt.show()
        
        return self._parameters

    def predict(self):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        self -- data set of examples you would like to label

        Returns:
        predictions -- predictions for the given dataset X
        """
                
        m = self._X.shape[1]
        n = len(self._parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.L_model_forward()

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        accuracy = np.sum((p == self._Y)/m)
           
        return p,accuracy



#Data loading
train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()
dims = [12288, 20, 7, 5, 1] 

#Class Initialization
dnn = DeepNN(train_x_orig, train_y_orig, dims,learning_rate = 0.0075, num_iterations = 2500)
print(dnn)
#Training
params = dnn.L_layer_model(print_cost = True)

# Train set predictions
train_predictions, train_accuracy = dnn.predict()
print('Training Accuracy:', train_accuracy)

#Test set predictions
dnn_test = DeepNN(test_x_orig, test_y_orig, dims, parameters = params)
test_predictions, test_accuracy = dnn_test.predict()
print('Test Accuracy:', test_accuracy)



