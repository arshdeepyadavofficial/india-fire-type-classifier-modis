# MODIS Fire Type Classification for India (2021-2023)

🔥 **Machine Learning Classification of Fire Types in India Using MODIS Satellite Data**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)
![MODIS](https://img.shields.io/badge/satellite-MODIS-red.svg)
![NASA](https://img.shields.io/badge/data-NASA%20FIRMS-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Overview

This project develops a comprehensive machine learning solution to classify different types of fire incidents across India using MODIS (Moderate Resolution Imaging Spectroradiometer) satellite data from NASA's Terra and Aqua satellites spanning 2021-2023. The system accurately distinguishes between various fire sources including forest fires, agricultural burning, volcanic activity, and other thermal anomalies to support environmental monitoring and disaster response efforts.

## 🎯 Project Objective

To develop a robust machine learning classification model that can accurately predict fire types using MODIS thermal anomaly data, enabling:
- **Timely disaster response** and emergency management
- **Environmental monitoring** and climate research
- **Resource management** for fire prevention
- **Agricultural burn detection** for policy enforcement
- **Forest fire management** for conservation efforts

## 📊 Dataset Information

### Data Sources
- **Primary Source**: NASA FIRMS (Fire Information for Resource Management System)
- **Geographic Coverage**: India
- **Temporal Coverage**: 2021-2023 (3 years of comprehensive data)
- **Satellites**: Terra (EOS AM) and Aqua (EOS PM) MODIS sensors
- **Spatial Resolution**: 1 km
- **Temporal Resolution**: 2-4 observations per day

### Dataset Files
```
├── modis_2021_India.csv    # MODIS fire data for 2021
├── modis_2022_India.csv    # MODIS fire data for 2022
└── modis_2023_India.csv    # MODIS fire data for 2023
```

## 🛰️ MODIS Technology Deep Dive

### Satellite Characteristics
- **Terra Satellite (EOS AM)**: Morning overpasses for consistent temporal coverage
- **Aqua Satellite (EOS PM)**: Afternoon overpasses for daily monitoring
- **Combined Coverage**: 2-4 observations per day in mid-latitudes like India
- **Launch Years**: Terra (1999), Aqua (2002) - proven long-term reliability

### Fire Detection Mechanism
- **Detection Method**: Contextual algorithms for thermal anomaly identification
- **Spectral Bands**: 
  - Mid-infrared channels (Bands 21/22) for active fire detection
  - Band 31 for surface temperature measurement
- **Classification System**: Each pixel categorized as missing, cloud, water, non-fire, fire, or unknown
- **Quality Assurance**: Built-in confidence metrics and validation protocols

### Key Parameters Analyzed
- **Geographic Coordinates**: Precise latitude/longitude positioning
- **Brightness Temperature**: Thermal intensity measurements (Kelvin)
- **Detection Confidence**: Algorithm certainty levels (0-100%)
- **Fire Radiative Power (FRP)**: Energy release measurements (MW)
- **Acquisition Metadata**: Timestamp, satellite source, and version tracking

## 🔬 Machine Learning Methodology

### Implemented Algorithms
1. **Logistic Regression**
   - Baseline linear classifier for binary/multiclass problems
   - Interpretable coefficients for feature importance
   - Fast training and prediction

2. **Random Forest**
   - Ensemble method combining multiple decision trees
   - Built-in feature importance ranking
   - Robust against overfitting
   - Handles missing values naturally

3. **K-Nearest Neighbors (KNN)**
   - Instance-based learning algorithm
   - Non-parametric approach
   - Effective for local pattern recognition

4. **XGBoost (Extreme Gradient Boosting)**
   - Advanced gradient boosting framework
   - Superior performance on structured data
   - Built-in regularization
   - Efficient handling of missing values

### Data Preprocessing Pipeline
- **Data Integration**: Seamless merging of multi-year datasets
- **Quality Control**: Missing value detection and handling
- **Feature Engineering**: Extraction of meaningful temporal and spatial patterns
- **Standardization**: Feature scaling for algorithm optimization
- **Encoding**: Categorical variable transformation
- **Feature Selection**: Statistical and mutual information-based selection

### Model Evaluation Framework
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-Validation**: Stratified k-fold for robust evaluation
- **Confusion Matrix**: Detailed classification performance analysis
- **Feature Importance**: Understanding key predictive variables
- **Model Comparison**: Systematic algorithm performance assessment

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Minimum 8GB RAM recommended
- 2GB free disk space for datasets

### Required Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/modis-fire-classification-india.git
cd modis-fire-classification-india
```

2. **Set Up Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Open the Main Notebook**
Navigate to `Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data.ipynb`

### Quick Start Example
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and combine datasets
df1 = pd.read_csv('modis_2021_India.csv')
df2 = pd.read_csv('modis_2022_India.csv') 
df3 = pd.read_csv('modis_2023_India.csv')
df = pd.concat([df1, df2, df3], ignore_index=True)

# Basic preprocessing and modeling
X = df.drop('type', axis=1)  # Features
y = df['type']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
```

## 📊 Exploratory Data Analysis Highlights

### Data Distribution Insights
- **Class Imbalance**: MODIS sensor data significantly outnumbers VIIRS observations
- **Confidence Patterns**: Bimodal distribution with peaks at high and low confidence levels
- **Temporal Trends**: Seasonal patterns correlating with agricultural cycles
- **Geographic Clustering**: Distinct fire hotspots across different Indian regions

### Key Findings
- **Agricultural Burning**: Concentrated in northern plains during post-harvest seasons
- **Forest Fires**: Prevalent in central and northeastern forested regions
- **Confidence Correlation**: Higher confidence detections show stronger spatial clustering
- **Seasonal Variability**: Peak fire activity during dry months (March-May, October-November)

## 📈 Real-World Applications

### Environmental Monitoring
- 🌡️ **Climate Change Research**: Long-term fire pattern analysis
- 🌿 **Ecosystem Health**: Forest degradation monitoring
- 🏭 **Air Quality**: Smoke and emission tracking
- 💧 **Water Resources**: Watershed impact assessment

### Disaster Management
- 🚨 **Early Warning Systems**: Real-time fire risk alerts
- 🚁 **Resource Allocation**: Strategic firefighting deployment
- 📍 **Evacuation Planning**: Population safety protocols
- 🗺️ **Risk Mapping**: Vulnerability assessment

### Policy and Governance
- 📋 **Agricultural Policies**: Crop residue burning regulations
- 🏛️ **Forest Management**: Conservation strategy development
- 🌾 **Sustainable Agriculture**: Alternative farming practice promotion
- 📊 **Impact Assessment**: Policy effectiveness evaluation

## 🗂️ Project Structure

```
modis-fire-classification-india/
├── 📁 data/
│   ├── modis_2021_India.csv              # 2021 fire incident data
│   ├── modis_2022_India.csv              # 2022 fire incident data
│   └── modis_2023_India.csv              # 2023 fire incident data
├── 📁 notebooks/
│   └── Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data.ipynb
├── 📁 results/
│   ├── 📁 model_performance/            # Accuracy metrics and reports
│   ├── 📁 visualizations/              # Generated plots and charts
│   └── 📁 feature_analysis/            # Feature importance studies
├── requirements.txt                     # Python dependencies
├── LICENSE                             # Project license
└── README.md                          # This file
```

## 🔍 Detailed Data Schema

### Target Variable
- **`type`**: Fire source classification (Primary target for ML models)
  - Categories: MODIS, VIIRS sensor classifications
  - Distribution: Imbalanced with MODIS predominance

### Feature Variables
| Variable | Type | Description | Range/Units |
|----------|------|-------------|-------------|
| `latitude` | Float | Geographic latitude coordinate | -90 to 90 degrees |
| `longitude` | Float | Geographic longitude coordinate | -180 to 180 degrees |
| `brightness` | Float | Brightness temperature | Kelvin |
| `confidence` | Integer | Detection confidence level | 0-100% |
| `frp` | Float | Fire Radiative Power | Megawatts (MW) |
| `satellite` | String | Source satellite identifier | Terra/Aqua |
| `instrument` | String | Detection instrument | MODIS/VIIRS |
| `version` | String | Data processing version | Version identifier |
| `acq_date` | Date | Acquisition date | YYYY-MM-DD |
| `acq_time` | Time | Acquisition time | HHMM (UTC) |

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Jupyter Notebook**: Interactive development environment
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing foundation
- **Scikit-learn**: Machine learning algorithms and tools

### Machine Learning Libraries
- **XGBoost**: Gradient boosting framework
- **Random Forest**: Ensemble learning methods
- **K-Nearest Neighbors**: Instance-based learning
- **Logistic Regression**: Linear classification models

### Visualization and Analysis
- **Matplotlib**: Static plotting and visualization
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive visualizations
- **Folium**: Geospatial mapping (if implemented)

### Development Tools
- **Git**: Version control
- **GitHub**: Repository hosting
- **VS Code**: Code editor integration
- **conda/pip**: Package management

## 📊 Performance Benchmarks

### Expected Model Performance
- **Random Forest**: ~85-90% accuracy on test data
- **XGBoost**: ~87-92% accuracy with optimal hyperparameters
- **Logistic Regression**: ~75-80% baseline performance
- **KNN**: ~80-85% accuracy with proper parameter tuning

### Key Success Metrics
- **Classification Accuracy**: Overall correct prediction rate
- **Precision by Class**: Fire type-specific accuracy
- **Recall Performance**: Detection completeness
- **F1-Score Balance**: Harmonic mean of precision and recall
- **Feature Importance**: Understanding predictive variables

## 🔗 References and Resources

### Scientific Publications
- [MODIS Collection 6 Active Fire Product User's Guide](https://modis.gsfc.nasa.gov/data/dataprod/mod14.php)
- [NASA Fire Information for Resource Management System](https://firms.modaps.eosdis.nasa.gov/)
- [Global Fire Emissions Database](https://www.globalfiredata.org/)

### Data Sources
- [NASA FIRMS Download Portal](https://firms.modaps.eosdis.nasa.gov/download/)
- [LP DAAC MODIS Products](https://lpdaac.usgs.gov/products/mod14a1v006/)
- [Earthdata NASA Portal](https://www.earthdata.nasa.gov/)

### Technical Documentation
- [MODIS Fire Detection Algorithm](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=04406)
- [Satellite Fire Detection Best Practices](https://www.publish.csiro.au/wf/WF19167)

## 📝 Citation

If you use this work in your research or projects, please cite:

```bibtex
@misc{modis_fire_classification_india_2025,
  title={MODIS Fire Type Classification for India (2021-2023): A Machine Learning Approach},
  author={Arshdeep Yadav},
  year={2025},
  publisher={GitHub},
  url={https://github.com/arshdeepyadavofficial/india-fire-type-classifier-modis},
  note={Machine learning classification of fire types using NASA MODIS satellite data}
}
```

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
1. **🐛 Bug Reports**: Submit issues for bugs or unexpected behavior
2. **💡 Feature Requests**: Suggest new functionality or improvements
3. **📖 Documentation**: Improve README, add tutorials, or enhance comments
4. **🔬 Research**: Add new algorithms or evaluation metrics
5. **🎨 Visualizations**: Create better charts and visual analyses

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines for Python code
- Add docstrings to all functions and classes
- Include unit tests for new functionality
- Update documentation for any changes
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Data Providers
- **NASA**: For providing free access to MODIS satellite data
- **FIRMS Team**: For maintaining the Fire Information Resource Management System
- **USGS**: For data processing and distribution infrastructure

### Scientific Community
- **MODIS Science Team**: For developing fire detection algorithms
- **Open Source Community**: For creating essential Python libraries
- **Research Contributors**: For advancing satellite-based fire monitoring

### Technical Support
- **Scikit-learn Developers**: For comprehensive ML tools
- **Pandas Team**: For powerful data manipulation capabilities
- **Jupyter Project**: For interactive computing environment

## 📧 Contact and Support

### Project Maintainer
- **GitHub**: [@arshdeepyadavofficial](https://github.com/arshdeepyadavofficial)
- **LinkedIn**: [LinkedIn](https://www.linkedin.com/in/arshdeep-yadav-827aa1257)

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️  for environmental monitoring and disaster management

**Keywords**: `machine-learning` `satellite-data` `fire-detection` `modis` `nasa` `india` `classification` `environmental-monitoring` `geospatial-analysis` `python` `scikit-learn` `xgboost` `random-forest` `remote-sensing` `disaster-management` `climate-change` `jupyter-notebook`

</div>
