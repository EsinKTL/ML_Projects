
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = load_breast_cancer()

X = data.data
y = data.target

#default
model = DecisionTreeClassifier()
model.fit(X, y)
print("Depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())

#max_depth = 3
model_depth3 = DecisionTreeClassifier(max_depth=3)
model_depth3.fit(X, y)
print("max_depth=3 → Depth:", model_depth3.get_depth(), "Leaves:", model_depth3.get_n_leaves())

# min_samples_split=10 → bir düğümü bölmek için en az 10 örnek gereksin
model_split10 =DecisionTreeClassifier(min_samples_split=10)
model_split10.fit(X, y)
print("min_samples_split=10 → Depth:", model_split10.get_depth(), "Leaves:", model_split10.get_n_leaves())

# criterion='entropy' → bölme kriteri gini yerine entropy olsun
model_entropy = DecisionTreeClassifier(criterion='entropy')
model_entropy.fit(X, y)
print("criterion=entropy → Depth:", model_entropy.get_depth(), "Leaves:", model_entropy.get_n_leaves())

# splitter='random' → en iyi bölme yerine rastgele bölme
model_random = DecisionTreeClassifier(splitter='random')
model_random.fit(X, y)
print("splitter=random → Depth:", model_random.get_depth(), "Leaves:", model_random.get_n_leaves())

plt.figure(figsize=(20,10))
plot_tree(model, feature_names= data.feature_names, class_names= data.target_names, filled= True)
plt.show()

path = model.cost_complexity_pruning_path(X,y)
alphas = path.ccp_alphas

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)
leaves_list = []
accuracies  = []
for alpha in alphas:
	model = DecisionTreeClassifier(ccp_alpha = alpha)
	model.fit(X_train, y_train)
	accuracies += [model.score(X_test, y_test)]
	leaves_list += [model.get_n_leaves()]
	
plt.figure(figsize=(20,10))
plt.title(f"ALPHAS VS LEAVES_LIST")
plt.plot(alphas, leaves_list)
plt.xlabel('alphas')
plt.ylabel('leaves')
plt.grid(True)
plt.show()

plt.figure(figsize=(20,10))
plt.title("ALPHAS VS ACCURACIES")
plt.plot(alphas, accuracies)
plt.xlabel('alphas')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()
	