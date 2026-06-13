import numpy as np
from scipy.stats import mode
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Default decision tree
model = DecisionTreeClassifier()
model.fit(X, y)
print("Depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())

# max_depth = 3
model_depth3 = DecisionTreeClassifier(max_depth=3)
model_depth3.fit(X, y)
print("max_depth=3 → Depth:", model_depth3.get_depth(), "Leaves:", model_depth3.get_n_leaves())

# min_samples_split=10 → require at least 10 samples to split a node
model_split10 = DecisionTreeClassifier(min_samples_split=10)
model_split10.fit(X, y)
print("min_samples_split=10 → Depth:", model_split10.get_depth(), "Leaves:", model_split10.get_n_leaves())

# criterion='entropy' → use entropy instead of gini
model_entropy = DecisionTreeClassifier(criterion='entropy')
model_entropy.fit(X, y)
print("criterion=entropy → Depth:", model_entropy.get_depth(), "Leaves:", model_entropy.get_n_leaves())

# splitter='random' → random split instead of best
model_random = DecisionTreeClassifier(splitter='random')
model_random.fit(X, y)
print("splitter=random → Depth:", model_random.get_depth(), "Leaves:", model_random.get_n_leaves())

# Visualize the default tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

# Cost-complexity pruning
path = model.cost_complexity_pruning_path(X, y)
alphas = path.ccp_alphas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
leaves_list = []
accuracies = []
for alpha in alphas:
    model = DecisionTreeClassifier(ccp_alpha=alpha)
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))
    leaves_list.append(model.get_n_leaves())

plt.figure(figsize=(20,10))
plt.title("ALPHAS VS LEAVES_LIST")
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

# Bagging (Bootstrap Aggregating)
B = 100
trees = []
for i in range(B):
    train_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train[train_idx]
    y_boot = y_train[train_idx]
    tree = DecisionTreeClassifier()   # new tree each iteration
    tree.fit(X_boot, y_boot)
    trees.append(tree)

predictions = np.array([tree.predict(X_test) for tree in trees])
y_pred_bagging = mode(predictions, axis=0).mode
accuracy_bagging = np.mean(y_pred_bagging == y_test)

# Single tree for comparison
single_tree = DecisionTreeClassifier()
single_tree.fit(X_train, y_train)
accuracy_single = single_tree.score(X_test, y_test)

print(f"Bagging accuracy: {accuracy_bagging}")
print(f"Single tree accuracy: {accuracy_single}")

# Compare on multiple datasets
datasets = {"Breast Cancer": load_breast_cancer(),
            "Wine": load_wine(),
            "Iris": load_iris()}

for name, data in datasets.items():
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Single Tree
    single_tree = DecisionTreeClassifier()
    single_tree.fit(X_train, y_train)
    acc_single = single_tree.score(X_test, y_test)

    # Bagging
    trees = []
    for i in range(100):
        train_idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train[train_idx]
        y_boot = y_train[train_idx]
        tree = DecisionTreeClassifier()
        tree.fit(X_boot, y_boot)
        trees.append(tree)
    predictions = np.array([tree.predict(X_test) for tree in trees])
    acc_bagging = np.mean(mode(predictions, axis=0).mode == y_test)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    acc_rf = rf.score(X_test, y_test)

    print(f"\n=== {name} ===")
    print(f"Single Tree   : {acc_single:.4f}")
    print(f"Bagging       : {acc_bagging:.4f}")
    print(f"Random Forest : {acc_rf:.4f}")

