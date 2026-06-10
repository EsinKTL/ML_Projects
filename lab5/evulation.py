import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import balanced_accuracy_score


def run_experiment(n, b, k, a):
	print(f"-------- WITH A = {a} VALUE --------")
	X = np.random.randn(n ,(5 + k))
	beta = np.concatenate([np.ones(5)*b, np.zeros(k)])
	linear_combination = a + X @ beta
	p = 1 / (1 + np.exp(-linear_combination))
	y = np.random.binomial(1,p,n)
	
	model_logreg = LogisticRegression()
	model_logreg.fit(X, y)
	y_pred_logreg = model_logreg.predict(X)
	error_logreg =	np.mean(y_pred_logreg != y)
	
	model_tree = DecisionTreeClassifier()
	model_tree.fit(X, y)
	y_pred_tree = model_tree.predict(X)
	error_tree = np.mean(y_pred_tree != y)
	
	print("Logistic Regression")
	print(f"Error: {error_logreg}")
	print(f"prediction: {y_pred_logreg}\n")
	print("Decision Tree")
	print(f"Error: {error_tree}")
	print(f"prediction: {y_pred_tree}\n")
	
	
	scores_logreg = cross_val_score(model_logreg, X, y, cv=10, scoring='accuracy')
	error_10logreg = 1 - scores_logreg.mean()
	
	scores_tree = cross_val_score(model_tree, X, y, cv=10, scoring='accuracy')
	error_10tree = 1 - scores_tree.mean()
	
	print("Logistic Regression with 10 fold cross validation")
	print(f"Error: {error_10logreg}")
	print("Decision Tree with 10 fold cross validation")
	print(f"Error: {error_10tree}")
	
	errorlistlog = []
	errorlisttree = []
	
	for i in range(100):
		train_idx = np.random.choice(n, n ,replace = True)
		test_idx = np.setdiff1d(np.arange(0,n), train_idx)
		X_train = X[train_idx]
		X_test = X[test_idx]
		y_train = y[train_idx]
		y_test = y[test_idx]
		model_logreg.fit(X_train, y_train)
		y_pred_logreg = model_logreg.predict(X_test)
		error_logreg = np.mean(y_pred_logreg != y_test)
		errorlistlog += [error_logreg]
		model_tree.fit(X_train, y_train)
		y_pred_tree = model_tree.predict(X_test)
		error_tree = np.mean(y_pred_tree != y_test)
		errorlisttree += [error_tree]
	
	errorstree = np.mean(errorlisttree)
	errorslog = np.mean(errorlistlog)
	print("Logistic Regression with bootstrap")
	print(f"Error: {errorslog}")
	print("Decision Tree with bootstrap")
	print(f"Error: {errorstree}")
	
	error_0632_log = 0.632 * errorslog + 0.368 * error_logreg
	error_0632_tree = 0.632 * errorstree + 0.368 * error_tree
	
	print("Logistic Regression with bootstrap 0.632")
	print(f"Error: {error_0632_log}")
	print("Decision Tree with bootstrap 0.632")
	print(f"Error: {error_0632_tree}")
	
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.5)
	
	model_logreg.fit(X_train, y_train)
	y_pred_logreg = model_logreg.predict_proba(X_test)[:,1]
	
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg)
	auc = roc_auc_score(y_test, y_pred_logreg)
	plt.title(f"ROC Curve (AUC = {auc:.2f})")
	
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.grid(True)
	plt.show()
	
	precision, recall, thresholds = precision_recall_curve(y_test, y_pred_logreg)
	plt.plot(recall, precision)
	plt.title('Precision-Recall Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.grid(True)
	plt.show()

run_experiment(1000,1,20,0)
run_experiment(1000,1,20,2)

n, b, k, a = 1000, 1, 20, 1
X = np.random.randn(n, 5 + k)
beta = np.concatenate([np.ones(5) * b, np.zeros(k)])
linear_combination = a + X @ beta
p = 1 / (1 + np.exp(-linear_combination))
y = np.random.binomial(1, p, n)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.5)
model_logreg = LogisticRegression()
model_logreg.fit(X_train, y_train)
y_proba = model_logreg.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0, 1, 100)
accuracies = []
balanced_accuracies = []
for t in thresholds:
	y_pred = (y_proba >= t).astype(int)
	accuracy = np.mean(y_pred == y_test)
	accuracies += [accuracy]
	balanced_accuracies += [balanced_accuracy_score(y_test, y_pred)]

t_balanced = y_train.mean()

plt.plot(thresholds, accuracies)
plt.axvline(x=0.5, color='red', linestyle='--', label = "t = 0.5")
plt.title('Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(thresholds, balanced_accuracies)
plt.axvline(x=t_balanced, color='red', linestyle='--', label = f"t = {t_balanced:.2f}")
plt.title('Balanced Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.grid(True)
plt.show()
	
	
	






