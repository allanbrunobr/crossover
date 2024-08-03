import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Step 1: Data Collection and Preprocessing

def load_and_sample_data():
    # Load datasets
    events = pd.read_csv('events.csv')
    item_properties_part1 = pd.read_csv('item_properties_part1.csv')
    item_properties_part2 = pd.read_csv('item_properties_part2.csv')
    category_tree = pd.read_csv('category_tree.csv')

    # Sample 5% of each dataset
    sample_fraction = 0.05
    random_seed = 42

    events_sample = events.sample(frac=sample_fraction, random_state=random_seed)
    item_properties_part1_sample = item_properties_part1.sample(frac=sample_fraction, random_state=random_seed)
    item_properties_part2_sample = item_properties_part2.sample(frac=sample_fraction, random_state=random_seed)

    # Combine sampled item properties datasets
    item_properties_sample = pd.concat([item_properties_part1_sample, item_properties_part2_sample], ignore_index=True)

    return events_sample, item_properties_sample, category_tree


def preprocess_data(events, item_properties):
    # Create user-item interaction matrix
    interactions = events[events['event'] == 'addtocart'].groupby(['visitorid', 'itemid']).size().reset_index(
        name='count')
    user_item_matrix = interactions.pivot_table(index='visitorid', columns='itemid', values='count', fill_value=0)

    # Concatenate item properties to form a text description
    item_properties['combined'] = item_properties['property'] + ' ' + item_properties['value']
    item_text = item_properties.groupby('itemid')['combined'].apply(lambda x: ' '.join(x)).reset_index()

    # Use TF-IDF to encode product descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(item_text['combined'])

    return user_item_matrix, tfidf_matrix, interactions


# Custom Dataset for PyTorch
class UserItemDataset(Dataset):
    def __init__(self, interactions):
        self.user_ids = torch.tensor(interactions['visitorid'].values, dtype=torch.long)
        self.item_ids = torch.tensor(interactions['itemid'].values, dtype=torch.long)
        self.labels = torch.tensor(interactions['count'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


# Step 2: Model Development

def train_svd(user_item_matrix):
    # Train SVD for collaborative filtering
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd_matrix = svd.fit_transform(user_item_matrix)
    return svd_matrix


def train_ncf(user_item_matrix, max_user_id, max_item_id, interactions, epochs=10, batch_size=64, latent_dim=20):
    # Neural Collaborative Filtering model
    class NeuralCollaborativeFiltering(nn.Module):
        def __init__(self, max_user_id, max_item_id, latent_dim):
            super(NeuralCollaborativeFiltering, self).__init__()
            self.user_embedding = nn.Embedding(max_user_id + 1, latent_dim)
            self.item_embedding = nn.Embedding(max_item_id + 1, latent_dim)
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, user_indices, item_indices):
            user_embedding = self.user_embedding(user_indices)
            item_embedding = self.item_embedding(item_indices)
            x = torch.cat([user_embedding, item_embedding], dim=-1)
            x = self.fc_layers(x)
            return x

    # Initialize model, loss function, and optimizer
    model = NeuralCollaborativeFiltering(max_user_id, max_item_id, latent_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare DataLoader
    dataset = UserItemDataset(interactions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for user_ids, item_ids, labels in dataloader:
            # Convert labels to binary (0 or 1)
            labels = (labels > 0).float()

            # Forward pass
            predictions = model(user_ids, item_ids).squeeze()
            loss = criterion(predictions, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    return model


# Step 3: Generate Recommendations

def recommend_items(user_item_matrix, svd_matrix, tfidf_matrix, user_id, top_n=5):
    # Compute cosine similarity for collaborative filtering
    user_similarities = cosine_similarity(svd_matrix)
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = np.argsort(-user_similarities[user_idx])[:top_n]

    # Compute content-based recommendations
    item_similarities = cosine_similarity(tfidf_matrix)
    recommended_items = set()

    for similar_user in similar_users:
        similar_user_id = user_item_matrix.index[similar_user]
        user_interactions = user_item_matrix.loc[similar_user_id]
        liked_items = user_interactions[user_interactions > 0].index

        for item_id in liked_items:
            if item_id not in recommended_items:
                similar_items = np.argsort(-item_similarities[item_id])[:top_n]
                recommended_items.update(similar_items)

    return list(recommended_items)


def save_recommendations(recommendations, filename='recommendations.csv'):
    # Save recommendations to CSV
    recommendations_df = pd.DataFrame(recommendations, columns=['ItemID'])
    recommendations_df.to_csv(filename, index=False)
    print(f"Recommendations saved to {filename}")


# Main script

def main():
    # Load and preprocess data
    events, item_properties, category_tree = load_and_sample_data()

    # Display category tree information
    print("Category Tree Sample:")
    print(category_tree.head())
    print(f"Category Tree Sample Shape: {category_tree.shape}")
    print("Unique Categories:", category_tree['categoryid'].nunique())  # Use 'categoryid' instead of 'category'

    user_item_matrix, tfidf_matrix, interactions = preprocess_data(events, item_properties)

    # Get max user and item IDs
    max_user_id = interactions['visitorid'].max()
    max_item_id = interactions['itemid'].max()
    print(f"Max User ID: {max_user_id}, Max Item ID: {max_item_id}")

    # Train models
    svd_matrix = train_svd(user_item_matrix)
    ncf_model = train_ncf(user_item_matrix, max_user_id, max_item_id, interactions)

    # Generate and save recommendations for all users
    recommendations_all_users = []
    for user_id in user_item_matrix.index:
        recommendations = recommend_items(user_item_matrix, svd_matrix, tfidf_matrix, user_id)
        recommendations_all_users.append((user_id, recommendations))

    # Save recommendations
    recommendations_df = pd.DataFrame(recommendations_all_users, columns=['UserID', 'RecommendedItems'])
    recommendations_df.to_csv('recommendations_all_users.csv', index=False)
    print("\nRecommendations for all users saved to recommendations_all_users.csv")


if __name__ == '__main__':
    main()
