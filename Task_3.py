import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("Social_Network_Ads.csv")

y = dataset['Age']
X = dataset[['EstimatedSalary', 'Purchased']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(X_train)
xtest = sc_x.transform(X_test)

print(xtrain[0:10, :])

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n", cm)

print("Accuracy : ", accuracy_score(y_test, y_pred))

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1,
                               stop = X_set[:, 0].max()+1, step=0.01),
                     np.arange(start=X_set[:, 1].min()-1,
                               stop=X_set[:, 1].max()+1, step=0.01))
plt.contourf(X1, X2, classifier.predict(
    np.array([X1.ravel(), X2.reval()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.show()