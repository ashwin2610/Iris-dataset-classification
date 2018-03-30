"""
MACHINE LEARNING SMP PROJECT-COMPARISON BETWEEN CLASSIFICATION ALGORITHMS

Libraries used:
Matplotlib
Numpy
Scikit-learn
"""

import numpy as np, matplotlib.pyplot as plt
from sklearn import *
iris=datasets.load_iris() #loading the dataset
X=iris.data
y=iris.target
#Creating test set containing 30% of the examples
X_test=X[105::,:]
y_test=y[105::]
X=X[0:105,:]
y=y[0:105]
plt.plot(X[:,0], X[:,1], 'ro')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
print
print

p=np.zeros((45))

def accuracy(y,p):
    acc=((y-p)==0)*1
    acc=(sum(acc)/45.0)*100
    return acc
    

#Logistic Regression
for C in range(1,31): #varying value of C from 0.1 to 3.0 with steps of 0.1
    logistic=linear_model.logistic.LogisticRegression(C=C/10.0)
    logistic.fit(X,y)
    h=logistic.predict_proba((X_test))
    for i in range(45):
        pred=h[i,:]
        ind=0
        for j in range(3):
            if pred[j]>pred[ind]:
                ind=j
        p[i]=ind
    print 'Accuracy of logistic regression with C = ',C/10.0, ':',
    print accuracy(y_test,p)

print
print

#Neural Network
for alpha in range(1,31): #varying alpha from 0.0001 to 0.0030
    nn=neural_network.multilayer_perceptron.MLPClassifier(alpha=alpha/10000.0)
    nn.fit(X,y)
    h1=nn.predict(X_test)
    print 'Accuracy of neural network with alpha = ', alpha/(10000.0), ': ', accuracy(y_test,h1)

print
print
#SVM
clf=svm.classes
for C in range(1,31): #varying value of C from 0.1 to 3.0 with steps of 0.1
    svm=clf.SVC(C=C/10.0) #kernel: gaussian radial basis kernel
    svm.fit(X,y)
    h2=svm.predict(X_test)
    print 'Accuracy of SVM with C = ', C/10.0, ': ', accuracy(y_test,h2)


"""
Maximum efficiency of each algorithm:
Logistic Regression:91.11% (C = 0.3)
Neural Netwroks: 95.56% (alpha = 0.001)
SVM: 97.78% (C = 0.2, C = 0.3, C= 0.4)
"""