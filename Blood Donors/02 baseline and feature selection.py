# -*- coding: utf-8 -*-
import pandas as pd

# Load the data for developing the model
train = pd.read_csv("./data/train.csv", index_col=0)


##############################################
## Magic happens here
##############################################

X = train.iloc[:,:-1]
y = train.iloc[:,-1]


# 1. Create features from the data / Preprocessing
from sklearn.preprocessing import MinMaxScaler

def transform(X):
    X_new = X.drop(["Total Volume Donated (c.c.)"], axis=1)
    X_new['Avg Donation'] = X['Total Volume Donated (c.c.)']/X['Number of Donations']
    X_new['Avg Wait'] = X['Months since First Donation']/X['Number of Donations']
    X_new['Month Ratio'] = X['Months since Last Donation']/X['Months since First Donation']
    X_new = MinMaxScaler().fit_transform(X_new)
    return X_new


X_new = transform(X)

# 2. Choose a model
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss

from sklearn.metrics import make_scorer

clf = LogisticRegressionCV(Cs=5
                           ,cv=10
                           , fit_intercept=False
                           , scoring=make_scorer(log_loss))


# 3. Valiate that model
from sklearn.model_selection import train_test_split

# Validation on only one fold
X_train, X_test, y_train, y_test = train_test_split(
                                                    X_new,y
                                                    ,random_state=123
                                                    , test_size=0.1)
clf.fit(X_train, y_train)


y_test_probs = clf.predict_proba(X_test)
y_test_preds = [y[0] for y in y_test_probs]
print("Log loss: ", log_loss(y_test, y_test_preds))


############################################
# Create prediction from submissions
############################################


X_val = pd.read_csv("./data/test.csv", index_col = 0)
X_val_new = transform(X_val)


y_probs = clf.predict_proba(X_val_new)
y_preds = [y[0] for y in y_probs]

submission = pd.DataFrame.from_dict(
        {"Made Donation in March 2007":y_preds}
        , dtype=float
        )

submission.set_index(X_val.index, inplace=True)
submission.to_csv("./out/demo.csv")