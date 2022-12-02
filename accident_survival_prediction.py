import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train = pd.read_csv('titanic_train.csv')
train.head()

# showing the columns with null value in yellow color
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.countplot(x='Survived', data=train, hue='Sex')

# plotting age of passengers based on different class to impute average of each pclass accordingly
sns.boxplot(x='Pclass', y='Age', data=train)


# data cleaning
# filling empty age field value with average according to pclasses
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)

# checking if some field have empty value
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# getting dummy variables
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

# deleting unwanted field to remove model complexity
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.drop('PassengerId', axis=1, inplace=True)

y = train['Survived']
X = train.drop('Survived', axis=1)

# splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# creating model
logmodel = LogisticRegression()
# training model
logmodel.fit(X_train, y_train)

# predicting values
predictions = logmodel.predict(X_test)

# checking accuracy with confusion matrix which is used to evaluate classificaton model
c_matrix = confusion_matrix(y_test, predictions)
print(c_matrix)

# classification report
c_report = classification_report(y_test, predictions)
print(c_report)
