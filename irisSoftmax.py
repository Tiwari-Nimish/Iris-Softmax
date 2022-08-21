import numpy as np

class SoftmaxReg():
    def __init__(self,lr=0.01,iters=1000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    def yLabel(self,y,c):
        y_label = np.zeros((len(y),c))

        i=0
        for rows in y_label:
            if(y[i]==0):
                y_label[i,0]=1
            if(y[i]==1):
                y_label[i,1]=1
            if(y[i]==2):
                y_label[i,2]=1
            i += 1
        
        return y_label

    def softmax(self,z):
        expX = np.exp(z)

        for i in range(len(z)):
            expX[i] /= np.sum(expX[i])
        
        return expX

    def fit(self,x,y,c):
        n_samples,n_features = x.shape
        self.weights = np.random.random((n_features,c))
        self.bias = np.random.random(c)

        for _ in range(self.iters):
            z = np.dot(x,self.weights) + self.bias
            y_predicted = self.softmax(z)
            y_new = self.yLabel(y,c)

            dw = (1/n_samples) * np.dot(x.T,y_predicted-y_new)
            db = (1/n_samples) * np.sum(y_predicted - y_new)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,x):
        z = np.dot(x,self.weights) + self.bias
        y_predicted = self.softmax(z)
        return np.argmax(y_predicted,axis=1)

    def accurate(self,y,y_predicted):
        return np.sum(y==y_predicted)/len(y)