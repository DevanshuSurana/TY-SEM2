# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
target_names = cancer.target_names

df = pd.DataFrame(data=cancer.data)
print(df.head())

# %%
scaler = StandardScaler()
scaler.fit(X)

# %%
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# %%
# Print the mean and standard deviation of each feature
print("Mean of each feature:")
print(scaler.mean_)
print("\nStandard Deviation of each feature:")
print(scaler.scale_)

# %%
param_grid = {'C': [0.01, 0.1, 1, 10, 100],'kernel': ['linear', 'rbf', 'poly']}


# %%
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)


# %%
pca = PCA(n_components=2)
grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_search.fit(pca.fit_transform(X_scaled), y)

# %%
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# %%
print("Best Params: ", best_params)

# %%
y_pred = best_model.predict(pca.transform(X_scaled))

# %%
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy with best model: {accuracy:.4f}")

# %%
colors = ['red' if label == 0 else 'blue' for label in y]
markers = ['o' if kernel == 'linear' else '^' if kernel == 'rbf' else 'x' for kernel in best_params['kernel']]

# %%
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# %%
# Plot the transformed data
colors = ['navy', 'darkorange']
markers = ['o', 's']
for target, color, marker in zip(np.unique(y), colors, markers):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], color=color, marker=marker, label=target_names[target])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Breast Cancer Dataset Visualization')
plt.legend(loc='upper right')
plt.show()

# %%
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = best_svm_classifier.predict(X_test_scaled)

# %%
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%
confusion_matrix(best_svm_classifier, X_test_scaled, y_test, cmap=plt.cm.Blues, display_labels=data.target_names)
plt.title('Confusion Matrix')
plt.show()


