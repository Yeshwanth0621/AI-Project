"""
Decision Tree Training for Worker Suitability
Trains a classifier to rank workers based on experience, safety, and availability
"""

import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


def load_data():
    """Load prepared decision tree data"""
    print("📂 Loading decision tree data...")
    
    features = np.load("processed_data/dt_features.npy")
    labels = np.load("processed_data/dt_labels.npy")
    
    print(f"✅ Loaded {len(features)} examples")
    print(f"   Features shape: {features.shape}")
    print(f"   Labels range: [{labels.min():.2f}, {labels.max():.2f}]")
    
    return features, labels


def train_decision_tree(X_train, y_train, X_val, y_val):
    """Train decision tree regressor"""
    print("\n🌳 Training decision tree...")
    
    # Use regressor since we have continuous suitability scores
    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"\n✅ Training complete!")
    print(f"   Train MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"   Val MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    # Feature importance
    print(f"\n📊 Feature Importance:")
    features_names = ['Experience Years', 'Safety Rating', 'Availability']
    for name, importance in zip(features_names, model.feature_importances_):
        print(f"   {name}: {importance:.3f}")
    
    return model


def save_model(model):
    """Save trained model"""
    print("\n💾 Saving model...")
    
    Path("models").mkdir(exist_ok=True)
    
    with open("models/decision_tree.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Saved to: models/decision_tree.pkl")


def main():
    print("=" * 60)
    print("Decision Tree Training")
    print("=" * 60)
    
    # Load data
    features, labels = load_data()
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=42
    )
    
    # Train
    model = train_decision_tree(X_train, y_train, X_val, y_val)
    
    # Save
    save_model(model)
    
    print("\n" + "=" * 60)
    print("✅ Decision tree training complete!")
    print("=" * 60)
    print("\nNext: Fine-tune SLM with: python train_worker_slm.py")


if __name__ == "__main__":
    main()
