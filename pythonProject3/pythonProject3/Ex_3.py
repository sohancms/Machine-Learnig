from sklearn.model_selection import train_test_split
from sklearn import datasets
from  sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.naive_bayes import GaussianNB
#load dataset
brest=datasets.load_breast_cancer()
x,y=brest.data,brest.target
#split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.2,random_state=42)
#linear model
lr_model=GaussianNB()
lr_model.fit(x_train,y_train)
y_pred=lr_model.predict(x_test).round()

print("confusion ")
print(confusion_matrix(y_test,y_pred))
print("\nclassification report:")
print(classification_report(y_test,y_pred,zero_division=1))
print("accuracy",accuracy_score(y_test,y_pred))

