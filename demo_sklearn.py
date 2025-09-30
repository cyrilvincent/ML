import sklearn
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms


print(sklearn.__version__)
#0 Set the seed
np.random.seed(42)

#1 Load data
dataframe = pd.read_csv("data/house/house.csv")

#2 Make dataset
y = dataframe["loyer"]
x = dataframe["surface"].values.reshape(-1, 1)


#3 Train Test Split
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

#4


#5 Creating the model
# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())

# f(x) = ax + b => 2 poids

#6 Fit
model.fit(xtrain, ytrain)

#7 Scoring (facultatif)
training_score= model.score(xtrain, ytrain)
testing_score = model.score(xtest, ytest)
print(f"Training Score: {training_score:.2f} Testing Score: {testing_score:.2f}")

#8 Predict
ypredicted = model.predict(x)

#9 Dataviz
plt.scatter(x, y)
xnew = np.arange(400)
ypredicted = model.predict(xnew.reshape(-1, 1))
plt.plot(xnew, ypredicted, color="red")
plt.show()
