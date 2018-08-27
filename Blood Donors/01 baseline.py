# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

# Load the data for developing the model
train = pd.read_csv("./data/train.csv", index_col=0)


##############################################
## Magic happens here
##############################################

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

X = train.iloc[:,:-1]
y = train.iloc[:,-1]

lr.fit(X,y)


# 3. Valiate that model
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=123)

lr.fit(X_train, y_train)

y_test_probs = lr.predict_proba(X_test)
y_test_preds = [y[0] for y in y_test_probs]


print("Log loss: ", log_loss(y_test, y_test_preds))



############################################
# Create prediction from submissions
############################################


X_val = pd.read_csv("./data/test.csv", index_col = 0)
y_preds = lr.predict(X_val)

submission = pd.DataFrame.from_dict(
        {"Made Donation in March 2007":y_preds}
        , dtype=float
        )

submission.set_index(X_val.index, inplace=True)
submission.to_csv("./out/demo.csv")