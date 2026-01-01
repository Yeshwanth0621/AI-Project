"""
Hybrid Worker Recommender
Combines TF-IDF similarity, decision tree ranking, and optional SLM reasoning
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import argparse


class HybridWorkerRecommender:
    """Hybrid recommendation system for worker allocation"""
    
    def __init__(self, use_slm=False):
        """
        Initialize recommender
        
        Args:
            use_slm: Whether to use SLM for reasoning (requires trained model)
        """
        print("🔧 Initializing Hybrid Worker Recommender...")
        
        # Load worker data
        self.workers_df = pd.read_csv("processed_data/workers_processed.csv")
        print(f"   Loaded {len(self.workers_df)} workers")
        
        # Load embeddings
        self.embeddings = np.load("processed_data/embeddings.npy")
        print(f"   Loaded embeddings: {self.embeddings.shape}")
        
        # Load vectorizer
        with open("processed_data/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"   Loaded TF-IDF vectorizer")
        
        # Load decision tree
        with open("models/decision_tree.pkl", "rb") as f:
            self.decision_tree = pickle.load(f)
        print(f"   Loaded decision tree model")
        
        # Optionally load SLM
        self.use_slm = use_slm
        if use_slm:
            try:
                from transformers import GPT2Tokenizer, GPT2LMHeadModel
                import torch
                
                self.tokenizer = GPT2Tokenizer.from_pretrained("worker_model")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.slm = GPT2LMHeadModel.from_pretrained(
                    "worker_model",
                    torch_dtype=torch.float32,
                    device_map="auto"
                )
                self.slm.eval()
                print(f"   Loaded SLM for reasoning generation")
            except Exception as e:
                print(f"   ⚠️  Could not load SLM: {e}")
                print(f"   Will use rule-based reasoning instead")
                self.use_slm = False
        
        print("✅ Recommender initialized!")
    
    def find_similar_workers(self, query: str, top_n=20, filters=None):
        """
        Stage 1: Find workers with similar skills using TF-IDF
        
        Args:
            query: Job requirement description
            top_n: Number of candidates to return
            filters: Dict with optional filters (min_experience, min_safety, only_available)
        
        Returns:
            List of worker indices sorted by similarity
        """
        # Expand query with synonyms and fix common typos
        expanded_query = self._expand_query(query)
        
        # Vectorize query
        query_vector = self.vectorizer.transform([expanded_query]).toarray()
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # Also check for job role match (boost if query contains role name)
        role_boost = self._get_role_boost(query)
        if role_boost is not None:
            for idx, row in self.workers_df.iterrows():
                if row['Job_Role'].lower() == role_boost.lower():
                    similarities[idx] += 0.3  # Boost matching roles
        
        # Get indices sorted by similarity
        similar_indices = np.argsort(similarities)[::-1]
        
        # Apply filters if provided
        if filters:
            filtered_indices = []
            filtered_sims = []
            for idx in similar_indices:
                worker = self.workers_df.iloc[idx]
                
                # Check filters
                if filters.get('min_experience', 0) > worker['Experience_Years']:
                    continue
                if filters.get('min_safety', 0) > worker['Safety_Rating']:
                    continue
                if filters.get('only_available', False) and worker['Availability'] != 'Available':
                    continue
                
                filtered_indices.append(idx)
                filtered_sims.append(similarities[idx])
                
                if len(filtered_indices) >= top_n:
                    break
            
            return filtered_indices[:top_n], filtered_sims[:top_n]
        
        return similar_indices[:top_n].tolist(), similarities[similar_indices[:top_n]].tolist()
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and fix typos"""
        query_lower = query.lower()
        
        # Common typos and corrections
        typo_fixes = {
            'devolper': 'developer',
            'devoloper': 'developer',
            'developper': 'developer',
            'devloper': 'developer',
            'elecrtical': 'electrical',
            'electriciann': 'electrician',
            'electrian': 'electrician',
            'managre': 'manager',
            'maneger': 'manager',
            'welder': 'welder welding TIG MIG ASME',
            'plumber': 'plumber plumbing pipe',
            'carpetner': 'carpenter',
            'mechaninc': 'mechanic',
            'safty': 'safety',
            'analys': 'analyst analysis',
        }
        
        # Synonyms to expand search
        synonyms = {
            'developer': 'developer python backend fastapi docker coding programming software engineer',
            'coder': 'developer python backend programming software engineer',
            'coding': 'developer python backend programming software',
            'programmer': 'developer python backend programming software engineer',
            'electrician': 'electrician electrical high-voltage NFPA PLC wiring',
            'wiring': 'electrician electrical wiring power',
            'manager': 'project manager PMP certified resource scheduling',
            'pm': 'project manager PMP certified',
            'safety': 'safety inspector audit incident lockout tagout',
            'analyst': 'data analyst power bi sql etl dashboards',
            'data': 'data analyst power bi sql etl pipelines',
            'hvac': 'hvac technician chiller epa 608 air conditioning',
            'forklift': 'forklift operator OSHA warehouse pallet toyota',
            'mechanic': 'mechanic diesel hydraulic engine maintenance preventive',
        }
        
        # Apply typo fixes
        for typo, fix in typo_fixes.items():
            if typo in query_lower:
                query_lower = query_lower.replace(typo, fix)
        
        # Expand with synonyms
        expanded_terms = [query_lower]
        for term, expansion in synonyms.items():
            if term in query_lower:
                expanded_terms.append(expansion)
        
        return ' '.join(expanded_terms)
    
    def _get_role_boost(self, query: str) -> str:
        """Check if query matches a specific job role"""
        query_lower = query.lower()
        
        role_keywords = {
            'Python Backend Developer': ['developer', 'python', 'backend', 'coder', 'coding', 'programming'],
            'Electrician': ['electrician', 'electrical', 'wiring', 'voltage'],
            'Welder': ['welder', 'welding', 'tig', 'mig'],
            'Project Manager': ['project manager', 'pmp', 'pm '],
            'Safety Inspector': ['safety', 'inspector', 'audit'],
            'Data Analyst': ['data analyst', 'analyst', 'bi ', 'sql'],
            'Carpenter': ['carpenter', 'carpentry', 'cabinet'],
            'HVAC Technician': ['hvac', 'air conditioning', 'chiller'],
            'Mechanic': ['mechanic', 'diesel', 'hydraulic', 'engine'],
            'Forklift Operator': ['forklift', 'operator', 'warehouse'],
            'General Laborer': ['laborer', 'cleanup', 'basic'],
        }
        
        for role, keywords in role_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    return role
        
        return None
    
    def rank_by_suitability(self, worker_indices):
        """
        Stage 2: Rank workers using decision tree
        
        Args:
            worker_indices: List of worker indices
            
        Returns:
            Sorted list of (index, suitability_score) tuples
        """
        rankings = []
        
        for idx in worker_indices:
            worker = self.workers_df.iloc[idx]
            
            # Prepare features for decision tree
            features = np.array([[
                worker['Experience_Years'],
                worker['Safety_Rating'],
                worker['Availability_Binary']
            ]])
            
            # Predict suitability
            suitability = self.decision_tree.predict(features)[0]
            rankings.append((idx, suitability))
        
        # Sort by suitability (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def generate_reasoning(self, query: str, worker_row, similarity_score, suitability_score):
        """
        Stage 3: Generate reasoning explanation
        
        Args:
            query: Original query
            worker_row: Pandas Series for worker
            similarity_score: Vector similarity score
            suitability_score: Decision tree suitability score
            
        Returns:
            Reasoning string
        """
        if self.use_slm and hasattr(self, 'slm'):
            # Use SLM to generate reasoning
            prompt = f"Query: {query}\nWorker: {worker_row['Name']} - {worker_row['Job_Role']}, {worker_row['Experience_Years']} years exp, {worker_row['Safety_Rating']:.1f} safety rating\nReason:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.slm.device) for k, v in inputs.items()}
            
            import torch
            with torch.no_grad():
                outputs = self.slm.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            reasoning = generated.split("Reason:")[-1].strip()
        else:
            # Rule-based reasoning
            reasons = []
            
            # Skills match
            if similarity_score > 0.3:
                reasons.append("strong skills match")
            elif similarity_score > 0.15:
                reasons.append("relevant skills")
            
            # Experience
            if worker_row['Experience_Years'] >= 15:
                reasons.append("highly experienced")
            elif worker_row['Experience_Years'] >= 10:
                reasons.append("experienced professional")
            elif worker_row['Experience_Years'] >= 5:
                reasons.append("good experience level")
            
            # Safety
            if worker_row['Safety_Rating'] >= 4.5:
                reasons.append("excellent safety record")
            elif worker_row['Safety_Rating'] >= 4.0:
                reasons.append("strong safety rating")
            
            # Availability
            if worker_row['Availability'] == 'Available':
                reasons.append("immediately available")
            
            reasoning = ", ".join(reasons).capitalize() if reasons else "Meets basic requirements"
        
        return reasoning
    
    def recommend(self, query: str, top_k=5, filters=None):
        """
        Full recommendation pipeline
        
        Args:
            query: Job requirement description
            top_k: Number of recommendations to return
            filters: Optional filters dict
            
        Returns:
            List of recommendation dicts
        """
        print(f"\n🔍 Query: {query}")
        if filters:
            print(f"   Filters: {filters}")
        
        # Stage 1: Vector similarity
        print("   Stage 1: Finding similar workers...")
        candidate_indices, similarities = self.find_similar_workers(query, top_n=20, filters=filters)
        print(f"   Found {len(candidate_indices)} candidates")
        
        # Stage 2: Decision tree ranking
        print("   Stage 2: Ranking by suitability...")
        rankings = self.rank_by_suitability(candidate_indices)
        
        # Stage 3: Generate reasoning for top-k
        print(f"   Stage 3: Generating reasoning for top {top_k}...")
        recommendations = []
        
        for idx, suit_score in rankings[:top_k]:
            worker = self.workers_df.iloc[idx]
            sim_score = similarities[candidate_indices.index(idx)]
            
            reasoning = self.generate_reasoning(query, worker, sim_score, suit_score)
            
            recommendations.append({
                'id': int(worker['ID']),
                'name': worker['Name'],
                'role': worker['Job_Role'],
                'experience': int(worker['Experience_Years']),
                'safety_rating': float(worker['Safety_Rating']),
                'availability': worker['Availability'],
                'skills': worker['Skills_Description'],
                'similarity_score': float(sim_score),
                'suitability_score': float(suit_score),
                'reasoning': reasoning
            })
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Worker Recommender")
    parser.add_argument("--query", type=str, help="Job requirement query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--use-slm", action="store_true", help="Use SLM for reasoning")
    parser.add_argument("--only-available", action="store_true", help="Only show available workers")
    parser.add_argument("--min-experience", type=int, default=0, help="Minimum years of experience")
    parser.add_argument("--min-safety", type=float, default=0.0, help="Minimum safety rating")
    
    args = parser.parse_args()
    
    # Initialize recommender
    recommender = HybridWorkerRecommender(use_slm=args.use_slm)
    
    # Build filters
    filters = {}
    if args.only_available:
        filters['only_available'] = True
    if args.min_experience > 0:
        filters['min_experience'] = args.min_experience
    if args.min_safety > 0:
        filters['min_safety'] = args.min_safety
    
    # Get query
    if args.query:
        query = args.query
    else:
        # Interactive mode
        print("\nInteractive Mode - Enter queries (or 'quit' to exit)")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Get recommendations
            recommendations = recommender.recommend(query, top_k=args.top_k, filters=filters or None)
            
            # Display
            print("\n" + "=" * 80)
            print(f"Top {len(recommendations)} Recommendations:")
            print("=" * 80)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['name']} - {rec['role']}")
                print(f"   Experience: {rec['experience']} years | Safety: {rec['safety_rating']:.2f} | {rec['availability']}")
                print(f"   Similarity: {rec['similarity_score']:.3f} | Suitability: {rec['suitability_score']:.3f}")
                print(f"   Reasoning: {rec['reasoning']}")
                print(f"   Skills: {rec['skills'][:100]}...")
        
        return
    
    # Single query mode
    recommendations = recommender.recommend(query, top_k=args.top_k, filters=filters or None)
    
    # Display
    print("\n" + "=" * 80)
    print(f"Top {len(recommendations)} Recommendations:")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']} - {rec['role']}")
        print(f"   Experience: {rec['experience']} years | Safety: {rec['safety_rating']:.2f} | {rec['availability']}")
        print(f"   Similarity: {rec['similarity_score']:.3f} | Suitability: {rec['suitability_score']:.3f}")
        print(f"   Reasoning: {rec['reasoning']}")
        print(f"   Skills: {rec['skills'][:100]}...")


if __name__ == "__main__":
    main()
