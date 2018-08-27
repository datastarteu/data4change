# -*- coding: utf-8 -*-


import pandas as pd

# Load the data for developing the model
train = pd.read_csv("./data/train.csv", index_col=0)
labels = pd.read_csv("./data/labels.csv", index_col=0)


##############################################
## Magic happens here
##############################################

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

X = train.iloc[:,:-1]
y = train.iloc[:,-1]

lr.fit(X,y)



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