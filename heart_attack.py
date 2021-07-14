import pandas as pd
import seaborn as sns

import statsmodels.api as st
from sklearn.linear_model import LogisticRegression
sns.set()
data = pd.read_csv('heart.csv')
data = data.copy()
print(data.describe())
x = data[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall']]
y = data['output']

x1 = st.add_constant(x)
reg = st.Logit(y,x1)
r=reg.fit()
print(r.summary())

model = LogisticRegression()
model.fit(x, y)
print(model.coef_,model.score(x, y))



