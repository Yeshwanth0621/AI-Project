"""
Worker Data Preparation Script
Loads CSV, generates embeddings, prepares training data
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
from pathlib import Path


def load_worker_data(csv_path: str = "workers_db.csv") -> pd.DataFrame:
    """Load and clean worker database"""
    print("📂 Loading worker data...")
    df = pd.read_csv(csv_path)
    
    # Clean data
    df['Skills_Description'] = df['Skills_Description'].fillna('')
    df['Availability_Binary'] = (df['Availability'] == 'Available').astype(int)
    
    print(f"✅ Loaded {len(df)} workers")
    print(f"   Job roles: {df['Job_Role'].nunique()} unique roles")
    print(f"   Available: {df['Availability_Binary'].sum()} workers")
    
    return df


def generate_embeddings(df: pd.DataFrame):
    """Generate TF-IDF vectors for skills descriptions"""
    print(f"\n🧮 Generating TF-IDF embeddings...")
    
    # Combine job role and skills for better embeddings
    texts = df.apply(
        lambda row: f"{row['Job_Role']} {row['Skills_Description']}", 
        axis=1
    ).tolist()
    
    # Use TF-IDF instead of sentence transformers (more stable, faster)
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),
        min_df=1
    )
    
    embeddings = vectorizer.fit_transform(texts).toarray()
    
    print(f"✅ Generated TF-IDF embeddings: shape {embeddings.shape}")
    
    return embeddings, vectorizer


def create_decision_tree_dataset(df: pd.DataFrame):
    """Create features for decision tree training"""
    print("\n📊 Creating decision tree dataset...")
    
    # Features for decision tree
    features = df[[
        'Experience_Years',
        'Safety_Rating',
        'Availability_Binary'
    ]].values
    
    # Create synthetic suitability scores based on criteria
    # This is simplified - in production you'd have labeled data
    suitability = (
        (df['Experience_Years'] / 20) * 0.4 +  # Experience weight: 40%
        (df['Safety_Rating'] / 5) * 0.4 +       # Safety weight: 40%
        df['Availability_Binary'] * 0.2         # Availability weight: 20%
    )
    
    print(f"✅ Created features: {features.shape}")
    print(f"   Mean suitability: {suitability.mean():.2f}")
    
    return features, suitability.values


def create_slm_training_data(df: pd.DataFrame, num_examples: int = 500):
    """Create training examples for SLM reasoning"""
    print(f"\n📝 Creating {num_examples} SLM training examples...")
    
    training_examples = []
    
    # Sample diverse job requirements
    job_requirements = {
        'Electrician': [
            "high-voltage maintenance",
            "PLC troubleshooting", 
            "NFPA 70E compliant",
            "industrial electrician"
        ],
        'Welder': [
            "TIG welding expert",
            "ASME Section IX certified",
            "aluminum welding",
            "stainless steel welding"
        ],
        'Project Manager': [
            "PMP certified",
            "resource allocation",
            "risk management"
        ],
        'Carpenter': [
            "finish carpentry",
            "OSHA 30 certified",
            "cabinet installation"
        ],
        'Safety Inspector': [
            "safety audits",
            "lockout/tagout training",
            "incident analysis"
        ],
        'Python Backend Developer': [
            "FastAPI experience",
            "PostgreSQL optimization",
            "Docker and CI/CD"
        ],
        'Data Analyst': [
            "Power BI dashboards",
            "SQL data modeling",
            "ETL pipelines"
        ],
        'HVAC Technician': [
            "EPA 608 certified",
            "chiller maintenance",
            "air balancing"
        ],
        'Mechanic': [
            "diesel engine diagnostics",
            "hydraulic systems",
            "preventive maintenance"
        ],
        'Forklift Operator': [
            "OSHA-compliant",
            "Toyota forklift certified",
            "warehouse safety"
        ]
    }
    
    # Generate training examples
    for idx, row in df.iterrows():
        role = row['Job_Role']
        
        if role not in job_requirements:
            continue
            
        # Pick a relevant requirement
        import random
        requirement = random.choice(job_requirements[role])
        
        # Create query
        query = f"Need {role.lower()} with {requirement}"
        
        # Worker details
        worker_info = f"{row['Name']} - {row['Job_Role']}, {row['Experience_Years']} years exp, {row['Safety_Rating']:.1f} safety rating, {row['Availability']}"
        
        # Generate reasoning based on skills match
        skills_lower = row['Skills_Description'].lower()
        
        reasons = []
        if requirement.lower() in skills_lower:
            reasons.append(f"specialized in {requirement}")
        if row['Experience_Years'] >= 10:
            reasons.append("highly experienced")
        elif row['Experience_Years'] >= 5:
            reasons.append("good experience level")
        if row['Safety_Rating'] >= 4.5:
            reasons.append("excellent safety record")
        elif row['Safety_Rating'] >= 4.0:
            reasons.append("strong safety rating")
        if row['Availability'] == 'Available':
            reasons.append("immediately available")
        
        if not reasons:
            reasons = ["relevant skills and qualifications"]
        
        reasoning = ", ".join(reasons)
        
        # Format as training example
        example = f"Query: {query}\nWorker: {worker_info}\nReason: {reasoning.capitalize()}"
        training_examples.append(example)
        
        if len(training_examples) >= num_examples:
            break
    
    print(f"✅ Created {len(training_examples)} training examples")
    
    return training_examples


def save_processed_data(df, embeddings, dt_features, dt_labels, slm_examples, vectorizer):
    """Save all processed data"""
    print("\n💾 Saving processed data...")
    
    # Create output directory
    Path("processed_data").mkdir(exist_ok=True)
    
    # Save worker dataframe with embeddings
    np.save("processed_data/embeddings.npy", embeddings)
    df.to_csv("processed_data/workers_processed.csv", index=False)
    
    # Save vectorizer
    with open("processed_data/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save decision tree data
    np.save("processed_data/dt_features.npy", dt_features)
    np.save("processed_data/dt_labels.npy", dt_labels)
    
    # Save SLM training data
    # Split into train and validation
    split_idx = int(len(slm_examples) * 0.8)
    train_examples = slm_examples[:split_idx]
    val_examples = slm_examples[split_idx:]
    
    with open("data/worker_train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    
    with open("data/worker_val.json", "w") as f:
        json.dump(val_examples, f, indent=2)
    
    print("✅ Saved all processed data:")
    print(f"   - Embeddings: processed_data/embeddings.npy")
    print(f"   - Workers: processed_data/workers_processed.csv")
    print(f"   - Vectorizer: processed_data/vectorizer.pkl")
    print(f"   - Decision tree: processed_data/dt_*.npy")
    print(f"   - SLM training: data/worker_train.json ({len(train_examples)} examples)")
    print(f"   - SLM validation: data/worker_val.json ({len(val_examples)} examples)")


def main():
    print("=" * 60)
    print("Worker Data Preparation")
    print("=" * 60)
    
    # Load data
    df = load_worker_data()
    
    # Generate embeddings
    embeddings, vectorizer = generate_embeddings(df)
    
    # Create decision tree dataset
    dt_features, dt_labels = create_decision_tree_dataset(df)
    
    # Create SLM training data
    slm_examples = create_slm_training_data(df, num_examples=400)
    
    # Save everything
    save_processed_data(df, embeddings, dt_features, dt_labels, slm_examples, vectorizer)
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train decision tree: python train_decision_tree.py")
    print("  2. Fine-tune SLM: python train_worker_slm.py")
    print("  3. Test recommender: python hybrid_recommender.py")


if __name__ == "__main__":
    main()
