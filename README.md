# EXP-1
```
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")

df.head()
df.info()

x = df.iloc[:,:-1].values
y= df.iloc[:,1].values
x
y

df.describe()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))



df1.describe()


X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)

y = df1.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

print("Size of X_train: ",len(X_train))

print(X_test)
print("Size of X_test: ",len(X_test))
```
## EXP-2
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self,learning_rate=0.1):
        self.learning_rate = learning_rate
        self._b = 0.0  #y-intercept
        self._w = None # weights assigned to input features
        self.misclassified_samples = []
    def fit(self, x: np.array, y: np.array, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []
        for _ in range(n_iter):
            # counter of the errors during this training interaction
            errors = 0
            for xi, yi in zip(x,y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)
    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b
    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0,1,-1)

df = pd.read_csv('IRIS.csv')
print(df.head())
# extract the label column
y = df.iloc[:,4].values
# extract features
x = df.iloc[:,0:3].values
#reduce dimensionality of the data
x = x[0:100, 0:2]
y = y[0:100]
#plot Iris Setosa samples
plt.scatter(x[:50,0], x[:50,1], color='orange', marker='o', label='Setosa')
#plot Iris Versicolour samples
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker='x', label='Versicolour')
#show the legend
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
#show the plot
plt.show()
#map the labels to a binary integer value
y = np.where(y == 'Iris-Setosa',1,-1)
x[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()
# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)
# train the model
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)
print("accuracy",accuracy_score(classifier.predict(x_test),y_test)*100)
# plot the number of errors during each iteration
plt.plot(range(1,len(classifier.misclassified_samples)+1),classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()
```
## EXP-3
```
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('IRIS.csv')
df.head

names = ['sepal-length','sepal-width','petal-length','petal-width','Class']

# Take first 4 columns ans assign them to variable "X"
X = df.iloc[:,0:4]
# Take first 5th columns and assign them to variable "Y". Object dtype refers to strings
Y = df.select_dtypes(include=[object])
X.head()
Y.head()

# Y actually contains all categories or classes
Y.species.unique()

# Now transforming categorial into numerical values
le = preprocessing.LabelEncoder()
Y = Y.apply(le.fit_transform)
Y.head()

# Train and test split (80% of data into training set and 20% into test data)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20)

# Feature Scaling
scaler = StandardScaler() 
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,Y_train.values.ravel())
predictions = mlp.predict(X_test)
print(predictions)

# Evaluation of algorithm performance in classifying flowers
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
```
## EXP-4
```
Name: Mirudhula D

Reg no.: 212221230060

import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

x=np.array([[0,0,1,1,],[0,1,0,1]])
y=np.array([[0,1,1,0]])
n_x=2
n_y=1
n_h=2
m=x.shape[1]
lr=0.1

np.random.seed(2)
w1=np.random.rand(n_h,n_x)
w2=np.random.rand(n_y,n_h)
losses=[]

def sigmoid(z):
    z=1/(1+np.exp(-z))
    return z
def forward_prop(w1,w2,x):
    z1=np.dot(w1,x)
    a1=sigmoid(z1)
    z2=np.dot(w2,a1)
    a2=sigmoid(z2)
    return z1,a1,z2,a2
    
def back_prop(m,w1,w2,z1,a1,z2,a2,y):
    dz2=a2-y
    dw2=np.dot(dz2,a1.T)/m
    dz1=np.dot(w2.T,dz2)*a1*(1-a1)
    dw1=np.dot(dz1,x.T)/m
    dw1=np.reshape(dw1,w1.shape)
    dw2=np.reshape(dw2,w2.shape)
    return dz2,dw2,dz1,dw1
    
iterations=10000
for i in range(iterations):
    z1,a1,z2,a2=forward_prop(w1,w2,x)
    loss=-(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
    w2=w2-lr*dw2
    w1=w1-lr*dw1
    
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")

def predict(w1,w2,input):
    z1,a1,z2,a2=forward_prop(w1,w2,test)
    a2=np.squeeze(a2)
    if a2>=0.5:
        print([i[0] for i in input],1)
    else:
        print([i[0] for i in input],0)

print("Input","Output")
test=np.array([[1],[0]])
predict(w1,w2,test)
test=np.array([[1],[1]])
predict(w1,w2,test)
test=np.array([[0],[1]])
predict(w1,w2,test)
test=np.array([[0],[0]])
predict(w1,w2,test)
```
## EXP-5
```
import numpy as np
import matplotlib.pyplot as plt
def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - landmark)**2)
def end_to_end(X1, X2, ys, mu1, mu2):
    from_1 = [gaussian_rbf(i, mu1) for i in zip(X1, X2)]
    from_2 = [gaussian_rbf(i, mu2) for i in zip(X1, X2)]
    # plot
    
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.scatter((x1[0], x1[3]), (x2[0], x2[3]), label="Class_0")
    plt.scatter((x1[1], x1[2]), (x2[1], x2[2]), label="Class_1")
    plt.xlabel("$X1$", fontsize=15)
    plt.ylabel("$X2$", fontsize=15)
    plt.title("Xor: Linearly Inseparable", fontsize=15)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(from_1[0], from_2[0], label="Class_0")
    plt.scatter(from_1[1], from_2[1], label="Class_1")
    plt.scatter(from_1[2], from_2[2], label="Class_1")
    plt.scatter(from_1[3], from_2[3], label="Class_0")
    plt.plot([0, 0.95], [0.95, 0], "k--")
    plt.annotate("Seperating hyperplane", xy=(0.4, 0.55), xytext=(0.55, 0.66),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel(f"$mu1$: {(mu1)}", fontsize=15)
    plt.ylabel(f"$mu2$: {(mu2)}", fontsize=15)
    plt.title("Transformed Inputs: Linearly Seperable", fontsize=15)
    plt.legend()

    # solving problem using matrices form
    # AW = Y
    A = []

    for i, j in zip(from_1, from_2):
        temp = []
        temp.append(i)
        temp.append(j)
        temp.append(1)
        A.append(temp)
    
    A = np.array(A)
    W = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(ys)
    print(np.round(A.dot(W)))
    print(ys)
    print(f"Weights: {W}")
    return W
def predict_matrix(point, weights):
    gaussian_rbf_0 = gaussian_rbf(np.array(point), mu1)
    gaussian_rbf_1 = gaussian_rbf(np.array(point), mu2)
    A = np.array([gaussian_rbf_0, gaussian_rbf_1, 1])
    return np.round(A.dot(weights))
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])

# centers
mu1 = np.array([0, 1])
mu2 = np.array([1, 0])

w = end_to_end(x1, x2, ys, mu1, mu2)

# TESTING:

print(f"Input:{np.array([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), w)}")
print(f"Input:{np.array([0, 1])}, Predicted: {predict_matrix(np.array([0, 1]), w)}")
print(f"Input:{np.array([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), w)}")
print(f"Input:{np.array([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), w)}")

mu1 = np.array([0, 0])
mu2 = np.array([1, 1])
w = end_to_end(x1, x2, ys, mu1, mu2)
print(f"Input:{np.array([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), w)}")
print(f"Input:{np.array([0, 1])}, Predicted: {predict_matrix(np.array([0, 1]), w)}")
print(f"Input:{np.array([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), w)}")
print(f"Input:{np.array([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), w)}")
```
## EXP-6
```
import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=pd.read_csv("/content/heart.csv")
X=data.iloc[:, :-1].values #features 
Y=data.iloc[:, -1].values  #labels 

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

mlp=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
training_loss=mlp.fit(X_train,y_train).loss_curve_

y_pred=mlp.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Losss")
plt.show()
```
