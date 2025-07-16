# 🔥 MODIS Fire Type Classification for India (2021-2023)

**Machine Learning Classification of Fire Types Using NASA MODIS Satellite Data**

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![MODIS](https://img.shields.io/badge/satellite-MODIS-red.svg)](https://modis.gsfc.nasa.gov/)
[![NASA](https://img.shields.io/badge/data-NASA%20FIRMS-blue.svg)](https://firms.modaps.eosdis.nasa.gov/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🌟 Overview

A machine learning solution to classify fire incidents across India using MODIS satellite data from NASA's Terra and Aqua satellites (2021-2023). Accurately distinguishes between forest fires, agricultural burning, and thermal anomalies for environmental monitoring and disaster response.

## 🎯 Objectives

- **🚨 Disaster Response**: Enable timely emergency management
- **🌍 Environmental Monitoring**: Track climate and ecosystem changes  
- **🔥 Fire Prevention**: Support resource management strategies
- **🌾 Agricultural Policy**: Detect crop residue burning
- **🌲 Forest Conservation**: Monitor protected areas

## 📊 Dataset

**Source**: NASA FIRMS (Fire Information Resource Management System)
- **Coverage**: India (2021-2023)
- **Satellites**: Terra & Aqua MODIS sensors
- **Resolution**: 1km spatial, 2-4 daily observations
- **Files**: `modis_2021_India.csv`, `modis_2022_India.csv`, `modis_2023_India.csv`

## 🛰️ MODIS Technology

- **Detection**: Contextual algorithms for thermal anomalies
- **Bands**: Mid-infrared (21/22) for fire, Band 31 for temperature
- **Features**: Latitude/longitude, brightness, confidence, Fire Radiative Power (FRP)
- **Quality**: Built-in confidence metrics (0-100%)

## 🤖 Machine Learning Models

| Algorithm | Description | Expected Accuracy |
|-----------|-------------|-------------------|
| **Random Forest** | Ensemble trees with feature importance | ~85-90% |
| **XGBoost** | Gradient boosting with regularization | ~87-92% |
| **Logistic Regression** | Linear baseline classifier | ~75-80% |
| **K-Nearest Neighbors** | Instance-based learning | ~80-85% |

### Pipeline
1. **Data Integration** → Merge multi-year datasets
2. **Preprocessing** → Handle missing values, feature scaling
3. **Feature Selection** → Statistical and mutual information methods
4. **Model Training** → Cross-validation with stratified splits
5. **Evaluation** → Accuracy, precision, recall, F1-score

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 8GB RAM
- 2GB disk space

### Installation
```bash
# Clone repository
git clone https://github.com/arshdeepyadavofficial/india-fire-type-classifier-modis.git
cd india-fire-type-classifier-modis

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter

# Launch notebook
jupyter notebook
```

### Usage Example
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df1 = pd.read_csv('modis_2021_India.csv')
df2 = pd.read_csv('modis_2022_India.csv') 
df3 = pd.read_csv('modis_2023_India.csv')
df = pd.concat([df1, df2, df3], ignore_index=True)

# Train model
X = df.drop('type', axis=1)
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

## 📈 Key Insights

- **Class Imbalance**: MODIS dominates over VIIRS observations
- **Temporal Patterns**: Peak activity during dry months (Mar-May, Oct-Nov)
- **Geographic Clusters**: Agricultural burning in northern plains, forest fires in central/northeast regions
- **Confidence Distribution**: Bimodal with high/low confidence peaks

## 🗂️ Project Structure

```
modis-fire-classification-india/
├── data/
│   ├── modis_2021_India.csv
│   ├── modis_2022_India.csv
│   └── modis_2023_India.csv
├── notebooks/
│   └── Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data.ipynb
├── results/
│   ├── model_performance/
│   └── visualizations/
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

**Core**: Python, Jupyter, Pandas, NumPy, Scikit-learn  
**ML**: XGBoost, Random Forest, KNN, Logistic Regression  
**Visualization**: Matplotlib, Seaborn

## � Data Schema

### Features
| Variable | Type | Description |
|----------|------|-------------|
| `latitude/longitude` | Float | Geographic coordinates |
| `brightness` | Float | Temperature (Kelvin) |
| `confidence` | Integer | Detection confidence (0-100%) |
| `frp` | Float | Fire Radiative Power (MW) |
| `satellite` | String | Terra/Aqua identifier |
| `type` | String | **Target variable** (MODIS/VIIRS) |

## 🔗 Resources

- [NASA FIRMS Portal](https://firms.modaps.eosdis.nasa.gov/)
- [MODIS Fire Product Guide](https://modis.gsfc.nasa.gov/data/dataprod/mod14.php)
- [LP DAAC MODIS Products](https://lpdaac.usgs.gov/products/mod14a1v006/)

## 📝 Citation

```bibtex
@misc{modis_fire_classification_india_2025,
  title={MODIS Fire Type Classification for India (2021-2023)},
  author={Arshdeep Yadav},
  year={2025},
  url={https://github.com/arshdeepyadavofficial/india-fire-type-classifier-modis}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **NASA FIRMS** for satellite data access
- **MODIS Science Team** for fire detection algorithms
- **Open Source Community** for ML tools

## 📧 Contact

**GitHub**: [@arshdeepyadavofficial](https://github.com/arshdeepyadavofficial)  
**LinkedIn**: [Arshdeep Yadav](https://www.linkedin.com/in/arshdeep-yadav-827aa1257)

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

Made with ❤️ for environmental monitoring and disaster management

</div>


