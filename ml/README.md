# Activity-Field Mapping SLM 🤖

A complete system for training a Small Language Model (SLM) to predict the best engineering field for any given activity.

## 🎯 Overview

This project fine-tunes **DistilGPT-2** using your JSONL dataset (204 examples) to create an intelligent activity-to-field mapping system. The model can identify the most relevant engineering field for any activity description and provide reasoning for its prediction.

## 📁 Project Structure

```
Test/
├── data.jsonl                          # Your training data (204 examples)
├── ml/
│   ├── config.yaml                     # Configuration settings
│   ├── requirements.txt                # Python dependencies
│   ├── prepare_data.py                 # Data preprocessing script
│   ├── train_model.py                  # Model training script
│   ├── inference.py                    # Inference and testing
│   ├── data/                           # Generated train/val splits
│   └── trained_model/                  # Saved model (after training)
├── backend/
│   └── ml_service.py                   # FastAPI service
└── frontend/
    └── activity_predictor.html         # Web interface
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd e:\Git\Test\ml
pip install -r requirements.txt
```

**Dependencies include:**
- `transformers` - Hugging Face library
- `torch` - PyTorch for deep learning
- `peft` - LoRA fine-tuning
- `datasets` - Data handling
- `fastapi` & `uvicorn` - API server
- `scikit-learn`, `matplotlib`, `yaml` - Utilities

### 2. Prepare Data

```bash
python prepare_data.py
```

This will:
- Load your `data.jsonl` file
- Format data for training
- Split into 80% train, 20% validation
- Save to `data/train.json` and `data/val.json`

### 3. Train the Model

```bash
python train_model.py
```

**Training Details:**
- **Base Model:** DistilGPT-2 (82M parameters)
- **Method:** LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Training Time:** ~5-15 minutes on CPU
- **Epochs:** 15
- **Output:** Model saved to `trained_model/`

**What to expect:**
```
================================================================
Activity-Field Mapping - Model Training
================================================================

📂 Loading train and validation datasets...
   Train examples: 163
   Val examples: 41

📥 Loading model: distilgpt2
🔧 Applying LoRA configuration...
trainable params: 294,912 || all params: 82,764,800 || trainable%: 0.3563

🚀 Starting training...
   Epochs: 15
   Batch size: 4
   Learning rate: 5e-05
```

### 4. Test the Model

```bash
# Test with sample activities
python inference.py --test

# Interactive testing
python inference.py --interactive

# Single prediction
python inference.py --activity "Design solar panel →"
```

### 5. Start the API Server

```bash
cd ..
python backend/ml_service.py
```

The API will start on `http://localhost:8000`

**API Endpoints:**
- `GET /` - Health check
- `GET /health` - Detailed status
- `POST /predict-field` - Single prediction
- `POST /batch-predict` - Batch predictions

**API Documentation:** http://localhost:8000/docs

### 6. Use the Web Interface

Open `frontend/activity_predictor.html` in your browser.

**Features:**
- Modern, responsive UI
- Real-time API status checking
- Pre-loaded example activities
- Instant predictions with field + reason

## 📊 Example Usage

### Python API

```python
from inference import ActivityFieldPredictor, load_config

config = load_config()
predictor = ActivityFieldPredictor("trained_model", config)

result = predictor.predict("Design 4-wheel rover chassis →")
print(f"Field: {result['field']}")
print(f"Reason: {result['reason']}")
```

### REST API

```bash
curl -X POST http://localhost:8000/predict-field \
  -H "Content-Type: application/json" \
  -d '{"activity": "Design solar panel efficiency optimization →"}'
```

**Response:**
```json
{
  "activity": "Design solar panel efficiency optimization →",
  "field": "renewables",
  "reason": "energy conversion",
  "confidence": 0.85
}
```

## ⚙️ Configuration

Edit `ml/config.yaml` to customize:

**Model Settings:**
```yaml
model:
  base_model: "distilgpt2"          # Base model name
  max_length: 128                   # Max sequence length
```

**Training Settings:**
```yaml
training:
  epochs: 15                         # Number of epochs
  batch_size: 4                      # Batch size
  learning_rate: 5.0e-5             # Learning rate
```

**LoRA Settings:**
```yaml
lora:
  r: 8                               # LoRA rank
  lora_alpha: 32                     # LoRA alpha
  lora_dropout: 0.1                  # Dropout rate
```

**Inference Settings:**
```yaml
inference:
  temperature: 0.7                   # Sampling temperature
  max_new_tokens: 50                 # Max generation length
  top_p: 0.9                         # Nucleus sampling
```

## 🎓 How It Works

### 1. Data Format

Your JSONL data is formatted as:
```json
{"prompt": "Design 4-wheel rover chassis →", "completion": "robotics (reason: autonomous mechanisms)"}
```

### 2. Training Process

- **LoRA Fine-Tuning:** Only trains 0.36% of parameters (294K out of 82M)
- **Causal Language Modeling:** Predicts next tokens given context
- **Format:** `Activity: [prompt] Field: [completion]`

### 3. Inference

Input: `"Activity: Design solar panel → Field:"`
Model generates: `"renewables (reason: energy conversion)"`

## 📈 Performance

With 204 training examples:
- **Training Loss:** Should decrease to < 1.0
- **Validation Accuracy:** Target > 85%
- **Inference Speed:** < 1 second per prediction
- **Model Size:** ~330 MB (base model + LoRA adapters)

## 🔧 Troubleshooting

### Issue: Model training is slow
**Solution:** Reduce batch_size to 2 or increase gradient_accumulation_steps

### Issue: Out of memory
**Solution:** Lower max_length to 64 or use smaller batch size

### Issue: API returns "Model not loaded"
**Solution:** Train the model first: `cd ml && python train_model.py`

### Issue: Poor predictions
**Solutions:**
- Increase training epochs (try 20-25)
- Lower learning rate (try 3e-5)
- Ensure data quality and consistency

## 🚀 Next Steps

**Improve the model:**
1. Collect more training data (aim for 500-1000 examples)
2. Use a larger base model (GPT-2 Small or TinyLlama)
3. Implement confidence scoring
4. Add few-shot examples for better generalization

**Extend functionality:**
1. Add user feedback loop to improve predictions
2. Implement caching for faster repeated queries
3. Create batch processing scripts
4. Deploy to cloud (Hugging Face Spaces, Railway, etc.)

## 📝 Notes

- **Dataset Size:** 204 examples is small but sufficient for this focused task
- **Model Choice:** DistilGPT-2 is perfect for CPU training and fast inference
- **Fine-Tuning Method:** LoRA enables efficient training without full model updates
- **Production Use:** Consider adding authentication and rate limiting for APIs

## 🤝 Contributing

To add more training data:
1. Add new examples to `data.jsonl` in the same format
2. Re-run `python prepare_data.py`
3. Re-train with `python train_model.py`

## 📄 License

This project uses open-source models and libraries. Check individual library licenses for details.

---

**Built with:** PyTorch, Transformers, PEFT, FastAPI
**Model:** DistilGPT-2 (Hugging Face)
**Training Method:** LoRA (Low-Rank Adaptation)
