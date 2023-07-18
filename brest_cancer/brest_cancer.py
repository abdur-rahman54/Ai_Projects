from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, accuracy_score

cancer1 = load_breast_cancer()
cancer = pd.DataFrame(cancer1.data, columns = cancer1.feature_names)
cancer.columns = cancer.columns.str.replace(' ','_')

# Add a column for the response variable: malignant or benign
cancer['Target'] = cancer1.target

# Take a look at the DataFrame again to double check we added the column properly
#print(cancer.shape)

# Select the first 10 columns of our DataFrame that we will use as the predictors in our models
x = cancer.iloc[:,:10]

# Select the response column 
y = cancer.Target

# Split these data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123) 

print("\nMethod 1:\n")

# Create the formula string 
all_columns = ' + '.join(cancer.columns[:10])
formula = "Target ~ " + all_columns
# Put the training predictors and responses into one DataFrame to be input into the model
trainingdata = pd.concat([x_train,y_train], axis = 1)

# Build the model
log_reg_1 = smf.logit(formula, data=trainingdata).fit()

# Predict responses 
pred_1 = log_reg_1.predict(x_test)
# round() rounds to nearest integer;
# 0.5 rounds to 0; 0.501 rounds to 1
prediction_1 = list(map(round, pred_1))

# Confusion matrix
cm = confusion_matrix(y_test, prediction_1) 
print ("\nConfusion Matrix : \n", cm) 

# Accuracy score
print('\nTest accuracy = ', accuracy_score(y_test, prediction_1))
print("")

print("\nMethod 2:\n")











