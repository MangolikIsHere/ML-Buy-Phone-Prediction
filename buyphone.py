from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data={ 'age':[25,35,45,20,30,50,40,55,60,35],
       'gender':['M','F','M','F','M','F','M','F','M','F'],
       'estimatedsalary':[30000,45000,60000,20000,35000,80000,90000,65000,80000,83000]  
      }
label_encoder= LabelEncoder()
df=pd.DataFrame(data)
df['gender']= label_encoder.fit_transform(df['gender'])
X = df
y= [0,1,0,1,0,1,0,1,0,1]
X_train , X_test , y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)
rf_classifier= RandomForestClassifier(n_estimators=100,random_state=42)
rf_classifier.fit(X_train,y_train)
y_pred=rf_classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))
age1=int(input("Enter your age:"))
gender1=input("Enter your gender (M/F):").upper()
salary1=int(input("Enter your salary:"))
gender_encoded=label_encoder.transform([gender1])[0]


user_data=[[age1,gender_encoded,salary1]]
prediction=rf_classifier.predict(user_data)
if prediction[0]==1:
    print("You're likely to purchase a smartphone")
else:
    print("You're not likely to purchase a smartphone")
