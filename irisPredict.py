import numpy as np
from sklearn.model_selection import train_test_split

from irisSoftmax import SoftmaxReg

FILE_NAME = "irisMod.data"
data = np.genfromtxt(FILE_NAME , delimiter=",")

regressor = SoftmaxReg()

n_sample,n_features = data.shape
n_features -= 1

x = data[:,0:n_features]
y = data[:,n_features]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)

regressor.fit(x_train,y_train,c=3)#'c' is the number of classes which are to be classified
y_pred = regressor.predict(x_test)
accur = regressor.accurate(y_test,y_pred)
print(accur)