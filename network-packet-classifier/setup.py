import nltk
import ssl
import os

def setup_environment():
    print("🛠️ Setting up Network Packet Classifier Environment")
    print("=" * 50)
    
    # Handle SSL for NLTK downloads
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_unverified_https_context = _create_unverified_https_context
    
    # Download NLTK data
    print("📥 Checking and downloading NLTK data...")
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        print("✅ NLTK data already available")
    except LookupError:
        print("📥 Downloading NLTK data (first time setup)...")
        nltk.download('stopwords')
        nltk.download('punkt')
        print("✅ NLTK data downloaded successfully")
    
    # Check if model needs training
    print("\n🔍 Checking model files...")
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        print("✅ Model files found")
    else:
        print("❌ Model files not found.")
        print("💡 Run: python train_model.py to train the model")
    
    print("\n🎉 Environment setup completed!")
    print("👉 Next steps:")
    print("   1. Train model: python train_model.py")
    print("   2. Run app: python app.py")
    print("   3. For real capture: run_as_admin.bat (as Administrator)")

if __name__ == "__main__":
    setup_environment()