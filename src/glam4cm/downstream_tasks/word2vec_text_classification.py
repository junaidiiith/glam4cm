import os
import fasttext
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim

# Step 4: Train a neural network classifier using the embeddings
class SimpleNN(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(SimpleNN, self).__init__()
		self.fc1 = nn.Linear(input_dim, 128)
		self.fc2 = nn.Linear(128, output_dim)
	
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)
		return x

# Step 2: Use the embeddings to transform the dataset
def get_embeddings(texts, model):
	embeddings = []
	for text in texts:
		words = text.split()
		word_vectors = [model.get_word_vector(word) for word in words]
		if word_vectors:
			embeddings.append(np.mean(word_vectors, axis=0))
		else:
			embeddings.append(np.zeros(model.vector_size))
	return np.array(embeddings)


def run(data):
    sentences = [text.split() for text in data['train_nodes'] + data['test_nodes']]    
    with open('data.txt', 'w') as f:
        f.write("\n".join(sentences))
        
    word2vec_model = fasttext.train_unsupervised(
        'data.txt', 
        dim=128, 
        ws=5, 
        minCount=1,
        epoch=100,
        lr=0.05, 
        workers=4
    )
    os.remove('data.txt')   

    X_train_embeddings = get_embeddings(data['train_nodes'], word2vec_model)
    X_test_embeddings = get_embeddings(data['test_nodes'], word2vec_model)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(data['train_node_classes'])
    y_test = label_encoder.transform(data['test_node_classes'])

    # Step 3: Train an SVM classifier using the embeddings
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_embeddings, y_train)
    y_pred_svm = svm_classifier.predict(X_test_embeddings)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))


    input_dim = X_train_embeddings.shape[1]
    output_dim = len(np.unique(y_train))

    model = SimpleNN(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_embeddings, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_embeddings, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the neural network
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        # Evaluate the neural network
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, y_pred_nn = torch.max(outputs, 1)
            y_pred_nn = y_pred_nn.numpy()
            print("Neural Network Classification Report:")
            report = classification_report(y_test, y_pred_nn)
		
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        print(f"Classification Report: {report}")
		