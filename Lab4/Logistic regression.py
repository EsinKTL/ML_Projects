import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = np.genfromtxt('earthquake.txt', dtype=str, skip_header=1)

popn = data[:, 0]
body = data[:, 1].astype(float)
surface = data[:, 2].astype(float)

# Numerical labels for sklearn: equake=0, explosn=1
le = LabelEncoder()
y = le.fit_transform(popn)  # automatically encodes in alphabetical order
X = np.column_stack([body, surface])

def scatterplot(popn, body, surface):
	plt.figure(figsize=(8, 6))
	
	# Dynamic filtering (masking) based on original string values in popn
	plt.scatter(body[popn == 'explosn'], surface[popn == 'explosn'],
	            color="green", label="Explosion")
	
	plt.scatter(body[popn == 'equake'], surface[popn == 'equake'],
	            color="pink", label="Earthquake")
	
	plt.xlabel('Body Wave Magnitude (body)')
	plt.ylabel('Surface Wave Magnitude (surface)')
	plt.title('Seismic Shocks Scatter Plot (Lab 4 - Task 1)')
	plt.legend()
	plt.grid(True)
	plt.show()

scatterplot(popn, body, surface)

def scatterplot_with_boundaries(popn, body, surface, model_no_reg, model_l2):
	plt.scatter(body[popn == 'explosn'], surface[popn == 'explosn'],
	            color='green', label='Explosion')
	plt.scatter(body[popn == 'equake'], surface[popn == 'equake'],
	            color='purple', edgecolors='gray', label='Earthquake')
	
	# Decision boundary: surface = (-β0 - β1*body) / β2
	body_range = np.linspace(body.min() - 0.5, body.max() + 0.5, 100)
	
	for model, color, linestyle, label in [
		(model_no_reg, 'blue', '-',  'No regularization'),
		(model_l2,     'red',  '--', 'L2 regularization'),
	]:
		b0 = model.intercept_[0]
		b1, b2 = model.coef_[0]
		boundary = (-b0 - b1 * body_range) / b2
		plt.plot(body_range, boundary, color=color,
		         linestyle=linestyle, linewidth=2, label=f'Decision boundary ({label})')
	
	plt.xlabel('Body Wave Magnitude (body)')
	plt.ylabel('Surface Wave Magnitude (surface)')
	plt.title('Seismic Shocks - Decision Boundaries')
	plt.legend()
	plt.grid(True)
	plt.show()

def log_likelihood(model, X, y):
	probs = model.predict_proba(X)
	# sum the log probability of the correct class for each observation
	ll = np.sum(np.log(probs[np.arange(len(y)), y] + 1e-15))
	return ll

model_no_reg = LogisticRegression(penalty=None)
model_no_reg.fit(X, y)

model_l2 = LogisticRegression(penalty='l2')
model_l2.fit(X, y)

print(" Without regularization")
print("Coefficients: ", model_no_reg.coef_)
print("Intercept: ", model_no_reg.intercept_)
print("Estimated predictions:\n ", model_no_reg.predict_proba(X))
print("Log-likelihood: ", log_likelihood(model_no_reg, X, y))

print(" With regularization")
print("Coefficients: ", model_l2.coef_)
print("Intercept: ", model_l2.intercept_)
print("Estimated predictions:\n ", model_l2.predict_proba(X))
print("Log-likelihood: ", log_likelihood(model_l2, X, y))

scatterplot(popn, body, surface)
scatterplot_with_boundaries(popn, body, surface, model_no_reg, model_l2)
