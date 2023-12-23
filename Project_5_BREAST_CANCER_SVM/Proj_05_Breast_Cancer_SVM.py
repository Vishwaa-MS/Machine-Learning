from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

bcancer = datasets.load_breast_cancer()
X = bcancer.data
y = bcancer.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
Xtrains, Xtest, Ytrains, Ytest = train_test_split(X, y, test_size=0.25, random_state=10)


#Linear SVM
svmc = SVC(kernel='linear')
svmc.fit(Xtrains, Ytrains)
Ypred1 = svmc.predict(Xtest)
svmcscore = accuracy_score(Ypred1, Ytest)
print('Accuracy score of Linear SVM Classifier is',100*svmcscore,'%\n')

# Kernel SVM RBF - Gaussian Kernal
ksvmc = SVC(kernel = 'rbf')
ksvmc.fit(Xtrains, Ytrains)
Ypred = ksvmc.predict(Xtest)
svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of Kernel SVM Classifier with RBF is',100*svmcscore,'%\n')

