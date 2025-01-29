from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


def run(data):
    X_train, X_test = data['train_nodes'], data['test_nodes']
    y_train, y_test = data['train_node_classes'], data['test_node_classes']

    pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'), verbose=True)

    print("Fitting SVM classifier")
    # Train the model
    pipeline.fit(X_train, y_train)

    print("Predicting")
    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))