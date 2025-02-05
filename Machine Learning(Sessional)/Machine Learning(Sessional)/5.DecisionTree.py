# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import graphviz

# Load sample dataset (Iris dataset)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report_output)

# Method 1: Visualizing the decision tree using plot_tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Method 2: Visualizing the decision tree using graphviz (for more detailed visualization)
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=iris.feature_names, 
                           class_names=iris.target_names, 
                           filled=True, rounded=True, special_characters=True)

# Display tree using graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves the tree as a PDF file
graph.view()  # Opens the PDF file
