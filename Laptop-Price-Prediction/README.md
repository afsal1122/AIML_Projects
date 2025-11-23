# 💻 Intelligent Laptop Price Predictor & Recommender

![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Machine Learning](https://img.shields.io/badge/AI-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Data Viz](https://img.shields.io/badge/Visualization-Plotly-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

An AI-powered application that estimates laptop prices in real-time using advanced machine learning algorithms. Analyze market trends, predict device costs based on specifications, and get personalized laptop recommendations through a modern, interactive dashboard.

## 🌟 Features

- **💰 Precision Price Prediction** - Estimate laptop prices based on CPU, RAM, Storage, GPU, and Brand.
- **📊 Interactive Market Explorer** - Visualize market trends, price distributions, and brand comparisons using Plotly.
- **🤖 Smart Recommendations** - Get tailored laptop suggestions based on your budget and usage requirements.
- **🔄 Automated ML Pipeline** - End-to-end data preprocessing, feature engineering, and model training.
- **📈 Comprehensive Metrics** - Detailed model evaluation including MAE, RMSE, and R² scores.
- **📱 Responsive Dashboard** - Clean, dark-themed UI built with Streamlit for seamless user experience.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PIP (Python Package Manager)

### Installation & Setup

1. **Clone the repository**
```bash
git clone [https://github.com/yourusername/Laptop-Price-Prediction.git](https://github.com/afsal1122/AIML_Projects/tree/main/Laptop-Price-Prediction)
cd Laptop-Price-Prediction
````

2.  **Create virtual environment**

<!-- end list -->

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3.  **Install dependencies**

<!-- end list -->

```bash
pip install -r requirements.txt
```

4.  **Preprocess Data**
    *Clean raw data and generate features.*

<!-- end list -->

```bash
python -m src.data.preprocess
```

5.  **Train Model**
    *Train the Random Forest algorithm and save model artifacts.*

<!-- end list -->

```bash
python -m src.models.train  
```

### Running the Application

1.  **Launch the Streamlit Dashboard**

<!-- end list -->

```bash
python -m streamlit run app/streamlit_app.py
```

2.  **Access the Interface**
    Open your browser and navigate to:
    `http://localhost:8501`

## 📁 Project Structure

```
Laptop-Price-Prediction/
├── app/
│   ├── pages/
│   │   ├── 1_Price_Predictor.py    # AI Prediction Interface
│   │   └── 2_Data_Explorer.py      # Market Visualization Dashboard
│   ├── app_utils.py                # Frontend utility functions
│   └── streamlit_app.py            # Main application entry point
├── data/
│   ├── raw/
│   │   └── training_dataset.csv    # Raw dataset
│   └── processed/
│       └── laptops_cleaned.csv     # Auto-generated processed data
├── models/                         # Serialized models (.pkl) & metrics
├── src/
│   ├── data/
│   │   ├── preprocess.py           # Data cleaning pipeline
│   │   └── features.py             # Feature engineering logic
│   ├── models/
│   │   ├── persistence.py          # Save/Load model logic
│   │   ├── train.py                # Model training script
│   │   └── evaluate.py             # Performance metric calculations
│   ├── recommend/
│   │   └── recommender.py          # Recommendation engine logic
│   └── utils.py                    # Global configurations & paths
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## 🏗️ Technical Architecture

### Machine Learning Pipeline

```
Raw Data → Preprocessing & Imputation → Feature Engineering → Model Training (RF) → Serialization
```

### System Components

  - **Frontend**: Streamlit (Multi-page application)
  - **Data Processing**: Pandas & NumPy for cleaning and manipulation
  - **ML Engine**: Scikit-Learn Pipeline with Random Forest Regressor
  - **Visualization**: Plotly Express & Graph Objects for interactive charts
  - **Persistence**: Joblib for efficient model saving/loading

## 🎯 Usage Modes

### 1\. Price Predictor 💲

  - **Input:** Select Brand, Processor (Intel/AMD/Apple), RAM, Storage, and GPU.
  - **Output:** Predicted market price with a confidence interval.
  - **Use Case:** Sellers setting prices or buyers checking fair market value.

### 2\. Market Explorer 📊

  - **Visuals:** Bar charts for Average Price by Brand, Scatter plots for Price vs. Specs.
  - **Filters:** Dynamic filtering by Brand, Screen Size, and Type.
  - **Use Case:** Analyzing current market trends and competitor analysis.

### 3\. Recommender System 🔍

  - **Input:** Budget range (Min/Max) and primary usage (Gaming, Work, Student).
  - **Output:** Top 5 laptop matches sorted by "Value Score".
  - **Logic:** Uses cosine similarity and weighted scoring based on specs.

## 📊 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **R² Score** | \~0.89 | Explains 89% of price variance |
| **MAE** | \~₹4,500 | Average absolute error in price |
| **RMSE** | \~₹6,200 | Root mean squared error |

*Note: Performance metrics may vary slightly based on the random seed and hyperparameter tuning results during `train.py` execution.*

## 🔧 Advanced Configuration

### Modifying the Dataset

To use your own dataset, place your CSV file in `data/raw/` and update the filename in `src/utils.py`:

```python
# src/utils.py
RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "your_new_dataset.csv")
```

### Tuning Hyperparameters

Edit `src/models/train.py` to adjust the search space for the model:

```python
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    # Add more parameters here
}
```

## 🐛 Troubleshooting

### Common Issues & Solutions

**ModuleNotFoundError: No module named 'src'**
Ensure you are running python commands from the root directory (`Laptop-Price-Prediction/`), not from inside `src/` or `app/`.

**Model File Not Found**
If the app complains about missing `.pkl` files, ensure you have run the training script first:

```bash
python src/models/train.py
```

**Streamlit Command Not Found**
Ensure your virtual environment is activated and requirements are installed.

```bash
pip install streamlit
```

## 🚀 Future Enhancements

  - [ ] Integration with live e-commerce APIs (Amazon/Flipkart) for real-time pricing.
  - [ ] Deep Learning integration (Neural Networks) for complex pattern recognition.
  - [ ] User authentication for saving favorite configurations.
  - [ ] Mobile-responsive layout optimization.
  - [ ] PDF Report generation for predicted analytics.

## 🤝 Contributing

We welcome contributions\! Please feel free to submit pull requests.

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/NewFeature`)
3.  Commit your changes (`git commit -m 'Add NewFeature'`)
4.  Push to the branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

## 📄 License

**This project is released for educational and research purposes.**

## 👥 Contributors

  - **[Afsal Rahiman T](https://github.com/afsal1122)** - Project Creator & Maintainer

-----

\<div align="center"\>
\<h3\>💻 Make Smart Tech Decisions with AI 💻\</h3\>
\<strong\>Accurate Predictions | Data-Driven Insights | Smart Choices\</strong\>
\</div\>

-----

**⭐ If you find this project helpful, please give it a star on GitHub\!**

```
```