import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler


data=pd.read_csv(r'C:\Users\venky\Downloads\data_banknote_authentication.txt')
print(data.head())

sns.countplot(x='Class',data=data)
plt.show()

scaler=StandardScaler()

scaler.fit(data.drop('Class',axis=1))

scaled_feature=scaler.fit_transform(data.drop('Class',axis=1))
df_feet=pd.DataFrame(scaled_feature,columns=data.columns[:-1])
print(df_feet.head())

 #train and spliting
x=df_feet
y=data['Class']


x=x.as_matrix()
y=y.as_matrix()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


#tensorflow contrib learn checking(DNN Classifier)
import tensorflow.contrib.learn.python.learn as learn

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)

classifier=learn.DNNClassifier(hidden_units=[10,20,10],n_classes=2,feature_columns=feature_columns)
classifier=classifier.fit(x_train,y_train,steps=200,batch_size=20)
predictions = list(classifier.predict(x_test, as_iterable=True))


#evaluation the model
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
Accuracy = metrics.accuracy_score(y_test, predictions)
print('Accuracy: {0:f}'.format(Accuracy))


#checking  RandomForestClassifier

from  sklearn.ensemble  import RandomForestClassifier

rfc=RandomForestClassifier()
fitting=rfc.fit(x_train,y_train)
rfc_prediction=rfc.predict(x_test)
print(classification_report(y_test,rfc_prediction))
print(confusion_matrix(y_test,rfc_prediction))
