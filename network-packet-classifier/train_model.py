import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import nltk
import ssl
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Handle SSL for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_unverified_https_context = _create_unverified_https_context

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK data for text processing...")
    nltk.download('stopwords')
    nltk.download('punkt')

class KaggleDataLoader:
    def __init__(self):
        self.datasets = {
            'unsw-nb15': 'mrwellsdavid/unsw-nb15',
            'cic-ids2017': 'cic-ids-2017', 
            'nsl-kdd': 'kashishparmar02/nslkdd'
        }
    
    def download_dataset(self, dataset_name='unsw-nb15'):
        """Download dataset from Kaggle"""
        try:
            # Setup Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            if dataset_name not in self.datasets:
                print(f"❌ Dataset {dataset_name} not supported")
                return None
            
            dataset_slug = self.datasets[dataset_name]
            download_path = f'kaggle_data/{dataset_name}'
            os.makedirs(download_path, exist_ok=True)
            
            print(f"📥 Downloading {dataset_name} from Kaggle...")
            api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
            
            print(f"✅ Dataset downloaded to: {download_path}")
            return download_path
            
        except Exception as e:
            print(f"❌ Kaggle download failed: {e}")
            print("💡 Make sure you have:")
            print("   - Kaggle API installed: pip install kaggle")
            print("   - Kaggle API key setup in ~/.kaggle/kaggle.json")
            return None
    
    def load_unsw_nb15(self, data_path):
        """Load and preprocess UNSW-NB15 dataset"""
        try:
            # Look for CSV files
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            if not csv_files:
                return None
            
            # Try to find the main training file
            train_file = None
            for file in csv_files:
                if 'training' in file.lower() or 'train' in file.lower():
                    train_file = os.path.join(data_path, file)
                    break
            
            if train_file is None:
                train_file = os.path.join(data_path, csv_files[0])
            
            print(f"📖 Loading data from: {train_file}")
            df = pd.read_csv(train_file)
            
            # UNSW-NB15 specific processing
            if 'label' in df.columns:
                # Create packet text representation from network features
                df['packet_text'] = df.apply(self.create_packet_text_unsw, axis=1)
                df['label'] = df['label'].apply(lambda x: 'malicious' if x == 1 else 'benign')
                
                print(f"📊 Loaded UNSW-NB15: {len(df)} records")
                print(f"📈 Class distribution: {df['label'].value_counts().to_dict()}")
                
                return df[['packet_text', 'label']].dropna()
            else:
                print("❌ 'label' column not found in dataset")
                return None
                
        except Exception as e:
            print(f"❌ Error loading UNSW-NB15: {e}")
            return None
    
    def create_packet_text_unsw(self, row):
        """Create text representation from UNSW-NB15 features"""
        try:
            features = []
            
            # Network features
            if 'proto' in row:
                features.append(f"proto_{str(row['proto']).lower()}")
            if 'service' in row:
                features.append(f"service_{str(row['service']).lower()}")
            if 'state' in row:
                features.append(f"state_{str(row['state']).lower()}")
            if 'srcip' in row:
                features.append(f"srcip_{str(row['srcip'])}")
            if 'dstip' in row:
                features.append(f"dstip_{str(row['dstip'])}")
            if 'sport' in row:
                features.append(f"sport_{str(row['sport'])}")
            if 'dsport' in row:
                features.append(f"dsport_{str(row['dsport'])}")
            
            # Protocol-specific features
            if 'TCP' in str(row.get('proto', '')):
                features.append("protocol_tcp")
            elif 'UDP' in str(row.get('proto', '')):
                features.append("protocol_udp")
            else:
                features.append("protocol_other")
            
            return ' '.join(features)
        except Exception as e:
            print(f"Warning: Error creating packet text: {e}")
            return "unknown_packet"
    
    def create_sample_dataset(self):
        """Create comprehensive sample dataset"""
        print("🔄 Creating enhanced sample dataset...")
        
        sample_data = []
        
        # Realistic network traffic patterns
        benign_patterns = [
            "GET /index.html HTTP/1.1 Host: example.com User-Agent: Mozilla/5.0",
            "POST /api/login HTTP/1.1 Content-Type: application/json",
            "TCP 192.168.1.100:54321 -> 93.184.216.34:80 Flags: PA",
            "UDP 192.168.1.101:12345 -> 8.8.8.8:53 DNS Query",
            "HTTPS 203.0.113.1:443 -> 192.168.1.50:65432 TLSv1.2",
            "ICMP 192.168.1.10 -> 8.8.8.8 Echo Request",
            "HTTP/1.1 200 OK Content-Type: text/html Server: nginx",
            "DNS Response: google.com Type: A Address: 8.8.8.8",
            "TCP 10.0.0.5:44322 -> 172.217.164.110:443 Flags: SA",
            "GET /static/css/style.css HTTP/1.1 Host: website.com"
        ]
        
        malicious_patterns = [
            "GET /admin/exec.php?cmd=whoami HTTP/1.1",
            "POST /shell.php HTTP/1.1 Content-Type: application/x-www-form-urlencoded",
            "TCP 10.0.0.100:4444 -> 192.168.1.1:4444 Flags: S",
            "GET /../../../etc/passwd HTTP/1.1",
            "POST /wp-admin/admin-ajax.php action=revslider_show_image HTTP/1.1",
            "TCP 192.168.1.200:1337 -> 10.0.0.1:22 Flags: S",
            "GET /cgi-bin/php?-d+allow_url_include=on HTTP/1.1",
            "UDP 10.0.0.77:31337 -> 192.168.1.255:31337",
            "POST /login.php username=admin password=admin HTTP/1.1",
            "GET /search?q=<script>alert('xss')</script> HTTP/1.1"
        ]
        
        # Generate balanced dataset
        for pattern in benign_patterns:
            for i in range(25):
                sample_data.append({
                    'packet_text': f"{pattern} Variant:{i}",
                    'label': 'benign'
                })
        
        for pattern in malicious_patterns:
            for i in range(25):
                sample_data.append({
                    'packet_text': f"{pattern} Variant:{i}",
                    'label': 'malicious'
                })
        
        df = pd.DataFrame(sample_data)
        print(f"✅ Created sample dataset with {len(df)} samples")
        return df

def train_model(use_kaggle_data=False, dataset_name='unsw-nb15'):
    """Train the ML model with optional Kaggle data"""
    print("🚀 Training Network Packet Classifier...")
    print("=" * 50)
    
    df = None
    data_source = "Kaggle" if use_kaggle_data else "Sample"
    
    if use_kaggle_data:
        print(f"📊 Using Kaggle dataset: {dataset_name}")
        loader = KaggleDataLoader()
        
        # Download and load dataset
        data_path = loader.download_dataset(dataset_name)
        
        if data_path and os.path.exists(data_path):
            if dataset_name == 'unsw-nb15':
                df = loader.load_unsw_nb15(data_path)
        
        if df is None or len(df) == 0:
            print("❌ Kaggle dataset loading failed, using sample data")
            df = loader.create_sample_dataset()
            data_source = "Sample (Fallback)"
    else:
        print("📊 Using sample dataset")
        loader = KaggleDataLoader()
        df = loader.create_sample_dataset()
    
    if df is None or len(df) == 0:
        print("❌ No data available for training")
        return None, None
    
    print(f"📈 Dataset Information:")
    print(f"   Source: {data_source}")
    print(f"   Total samples: {len(df):,}")
    print(f"   Class distribution:")
    print(df['label'].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['packet_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"\n🔤 Vectorizing text data...")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Vectorize text with enhanced features
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"   Feature space: {X_train_vec.shape[1]} dimensions")
    
    # Train model with optimized parameters
    print(f"\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    print(f"\n📊 Evaluating model performance...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    
    # Feature importance analysis
    print(f"\n🔝 Top 10 Most Important Features:")
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i:2d}. {feature_names[idx]:<20} {importances[idx]:.4f}")
    
    # Save model and vectorizer
    print(f"\n💾 Saving model files...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print(f"✅ Model saved as 'model.pkl'")
    print(f"✅ Vectorizer saved as 'vectorizer.pkl'")
    
    # Model size info
    model_size = os.path.getsize('model.pkl') / (1024 * 1024)
    vectorizer_size = os.path.getsize('vectorizer.pkl') / (1024 * 1024)
    
    print(f"📦 Model size: {model_size:.2f} MB")
    print(f"📦 Vectorizer size: {vectorizer_size:.2f} MB")
    
    return model, vectorizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Network Packet Classifier')
    parser.add_argument('--kaggle', action='store_true', 
                       help='Use Kaggle dataset (requires Kaggle API setup)')
    parser.add_argument('--dataset', default='unsw-nb15', 
                       choices=['unsw-nb15', 'cic-ids2017', 'nsl-kdd'],
                       help='Kaggle dataset to use')
    
    args = parser.parse_args()
    
    print("🛡️  Network Packet Classifier - Model Training")
    print("=" * 50)
    
    if args.kaggle:
        print("🎯 Mode: Kaggle Dataset Training")
        print(f"📚 Dataset: {args.dataset}")
        print("💡 Note: This requires Kaggle API setup")
    else:
        print("🎯 Mode: Sample Data Training")
        print("💡 Tip: Use --kaggle flag for better accuracy with real datasets")
    
    print("=" * 50)
    
    model, vectorizer = train_model(
        use_kaggle_data=args.kaggle, 
        dataset_name=args.dataset
    )
    
    if model is not None:
        print("\n🎉 Training completed successfully!")
        print("👉 Next step: Run 'python app.py' to start real-time classification")
        print("💡 For real packet capture: Run 'run_as_admin.bat' as Administrator")
    else:
        print("\n❌ Training failed!")
        print("💡 Troubleshooting tips:")
        print("   - Check if all dependencies are installed")
        print("   - For Kaggle: Ensure API is properly configured")
        print("   - Check internet connection for dataset download")