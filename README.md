# 🔥 MODIS Fire Type Classification for India (2021-2023)

**Machine Learning Classification of Fire Types Using MODIS Satellite Data**

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-webapp-red.svg)](https://streamlit.io/)
[![MODIS](https://img.shields.io/badge/satellite-MODIS-red.svg)](https://modis.gsfc.nasa.gov/)
[![NASA](https://img.shields.io/badge/data-NASA%20FIRMS-blue.svg)](https://firms.modaps.eosdis.nasa.gov/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### **🔥 [DOWNLOAD BEST FIRE DETECTION MODEL](https://drive.google.com/your-model-link-here)**
## 📋 Project Overview

This project develops a comprehensive machine learning system to classify fire types in India using MODIS satellite data from 2021-2023. The solution includes data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive web application.

## 🎯 Objectives

- **Primary**: Develop accurate fire type classification using MODIS thermal and geographic features
- **Secondary**: Create deployable web application for real-time fire type prediction
- **Impact**: Support environmental monitoring and disaster management initiatives

## 📊 Dataset

**Source**: NASA FIRMS (Fire Information Resource Management System)
- **Coverage**: India, 2021-2023 (3 years)
- **Satellites**: Terra & Aqua MODIS sensors
- **Resolution**: 1 km spatial resolution
- **Size**: 500,000+ fire detection records
- **Format**: 3 CSV files (annual datasets)

### Key Features
- **Geographic**: latitude, longitude coordinates
- **Thermal**: brightness, bright_t31, frp (Fire Radiative Power)
- **Sensor**: scan, track, confidence levels (0-100%)
- **Temporal**: acquisition date, time, day/night flag
- **Metadata**: satellite, instrument type
- **Target**: fire type classification (MODIS/VIIRS)

## 🛠️ Technical Implementation

### Data Processing Pipeline
- **Data Integration**: Merged multi-year datasets with validation
- **Quality Assurance**: Missing value analysis, duplicate detection, outlier treatment
- **Feature Engineering**: Temporal extraction (hour, month, season), categorical encoding
- **Data Standardization**: StandardScaler normalization for model consistency
- **Class Balancing**: SMOTE implementation for imbalanced dataset handling

### Machine Learning Models
Implemented and evaluated multiple classification algorithms:
- **Logistic Regression**: Linear baseline model with good interpretability
- **Decision Tree**: Non-linear decision boundaries with feature importance
- **Random Forest**: Ensemble method achieving 99.9%+ accuracy (Selected Model)
- **K-Nearest Neighbors**: Instance-based learning approach

### Model Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-Validation**: Robust model validation with confusion matrices
- **Feature Importance**: Analysis of key predictive features
- **Model Comparison**: Comprehensive performance benchmarking

## 📈 Key Results

### Model Performance
- **Best Model**: Random Forest Classifier
- **Accuracy**: 97.77%+ t
- **Precision/Recall**: High performance across all fire type classes
- **Feature Importance**: Thermal features (brightness, FRP) most predictive

### Data Insights
- **Temporal Patterns**: Clear seasonal fire detection trends
- **Geographic Distribution**: Regional clustering across Indian subcontinent
- **Confidence Levels**: Bimodal distribution indicating detection certainty
- **Class Distribution**: Significant imbalance requiring SMOTE correction

## 🎨 Visualization Features

### Comprehensive Analytics
- **Fire-Themed Color Schemes**: Custom palettes for consistent branding
- **Interactive Geographic Maps**: Folium-based visualization with 5000+ fire points
- **Statistical Distributions**: Histograms, box plots, correlation heatmaps
- **Temporal Analysis**: Monthly trends, seasonal patterns, hourly distributions
- **Model Performance**: Confusion matrices, accuracy comparisons, feature importance

### Interactive Elements
- **Clickable Maps**: Detailed fire information popups
- **Multi-Layer Visualization**: Satellite imagery, street maps, terrain views
- **Real-Time Updates**: Dynamic filtering and zoom capabilities
- **Professional Styling**: Publication-ready plots with enhanced aesthetics

## 🚀 Web Application

### Streamlit Deployment
- **User Interface**: Professional fire-themed design with gradient backgrounds
- **Input Features**: Interactive forms for all model parameters
- **Real-Time Prediction**: Instant fire type classification with confidence scores
- **Responsive Design**: Mobile-friendly interface with custom CSS styling

### Application Features
- **Parameter Validation**: Min/max constraints with error handling
- **Loading Animations**: User experience enhancements
- **Color-Coded Results**: Visual fire type classification output
- **Detailed Descriptions**: Comprehensive fire type explanations
- **Professional Footer**: Developer attribution and contact links

## 📁 Project Structure

```
india-fire-type-classifier-modis/
├── 📁 data/                          # Raw and processed datasets
├── 📓 Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data.ipynb      # Main analysis notebook
├── 🐍 app.py                         # Streamlit web application
├── 💾 models/                        # Trained models and scalers
├── 📊 visualizations/               # Generated plots and maps
├── 📄 README.md                     # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Git

### Dependencies
- **Core**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, folium
- **ML**: imblearn, joblib
- **Web App**: streamlit
- **Utilities**: datetime, warnings

### Quick Start
1. Clone repository and navigate to project directory
2. Install required dependencies
3. Open Jupyter notebook for analysis
4. Run Streamlit app for web interface

## 🎯 Use Cases & Applications

### Environmental Monitoring
- **Wildfire Detection**: Early warning systems for forest fires
- **Agricultural Monitoring**: Crop burning detection and analysis
- **Urban Planning**: Heat island effect and urban fire risk assessment

### Disaster Management
- **Emergency Response**: Rapid fire type classification for resource allocation
- **Risk Assessment**: Historical fire pattern analysis for prevention
- **Policy Support**: Data-driven environmental policy recommendations

### Research Applications
- **Climate Studies**: Fire pattern correlation with weather data
- **Ecological Research**: Impact assessment on biodiversity
- **Remote Sensing**: Advanced satellite data processing techniques

## 🏆 Technical Achievements

### Data Science Excellence
- **End-to-End Pipeline**: Complete ML workflow from raw data to deployment
- **Advanced Preprocessing**: Comprehensive data cleaning and feature engineering
- **Model Optimization**: Systematic algorithm comparison and selection
- **Production Ready**: Scalable and maintainable code architecture

### Innovation Highlights
- **Interactive Deployment**: User-friendly web application interface
- **Geographic Intelligence**: Spatial analysis with interactive mapping
- **Custom Visualizations**: Fire-themed design with professional aesthetics
- **Real-World Impact**: Practical application for environmental monitoring

## 🔬 Future Enhancements

### Technical Improvements
- **Deep Learning**: CNN/RNN implementation for enhanced accuracy
- **Time Series Forecasting**: Predictive fire occurrence modeling
- **API Development**: RESTful services for system integration
- **Cloud Deployment**: Scalable AWS/Azure infrastructure

### Feature Expansion
- **Real-Time Processing**: Live satellite data stream integration
- **Mobile Application**: Cross-platform mobile app development
- **Advanced Analytics**: Multi-temporal analysis and trend prediction
- **Integration Capabilities**: Weather data fusion for enhanced predictions

## 📚 References & Data Sources

- [NASA FIRMS Portal](https://firms.modaps.eosdis.nasa.gov/)
- [MODIS Fire Product Documentation](https://modis.gsfc.nasa.gov/data/dataprod/mod14.php)
- [LP DAAC MODIS Products](https://lpdaac.usgs.gov/products/mod14a1v006/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📝 Citation

```bibtex
@misc{modis_fire_classification_india_2025,
  title={MODIS Fire Type Classification for India (2021-2023): 
         Machine Learning Approach for Satellite-Based Fire Detection},
  author={Arshdeep Yadav},
  year={2025},
  url={https://github.com/arshdeepyadavofficial/india-fire-type-classifier-modis},
  note={Machine Learning classification system using NASA MODIS satellite data}
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA FIRMS** for comprehensive satellite fire data access
- **MODIS Science Team** for advanced fire detection algorithms
- **Open Source Community** for machine learning tools and libraries
- **Streamlit Team** for intuitive web application framework

## 📧 Contact & Support

**Developer**: Arshdeep Yadav  
**GitHub**: [arshdeepyadavofficial](https://github.com/arshdeepyadavofficial)  
**LinkedIn**: [Arshdeep Yadav](https://www.linkedin.com/in/arshdeep-yadav-827aa1257)  
**Email**: Available through GitHub profile

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

*Made with ❤️ for environmental monitoring and disaster management*

**Empowering data-driven decisions for a safer, sustainable future**

</div>



