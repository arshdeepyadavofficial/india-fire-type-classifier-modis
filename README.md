# 🔥 MODIS Fire Type Classification for India (2021-2023)

**Machine Learning Classification of Fire Types Using NASA MODIS Satellite Data**

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![MODIS](https://img.shields.io/badge/satellite-MODIS-red.svg)](https://modis.gsfc.nasa.gov/)
[![NASA](https://img.shields.io/badge/data-NASA%20FIRMS-blue.svg)](https://firms.modaps.eosdis.nasa.gov/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Project Overview
This project aims to develop a machine learning classification model that can accurately predict the type of fire using MODIS fire detection data for India from 2021 to 2023. The project focuses on distinguishing between different fire sources such as vegetation fires, volcanic activity, and other thermal anomalies using satellite-captured features.

## 🎯 Objective
Develop a classification model to distinguish between different fire types (MODIS/VIIRS) using satellite thermal and geographic features.

## 📊 Dataset

**Source**: NASA FIRMS (Fire Information Resource Management System)
- **Time Period**: 2021-2023
- **Geographic Coverage**: India  
- **Satellites**: Terra & Aqua MODIS sensors
- **Spatial Resolution**: 1 km
- **Files**: 3 CSV files (one per year)

### Key Features
- **Geographic**: latitude, longitude
- **Thermal**: brightness, bright_t31, frp (Fire Radiative Power)
- **Sensor**: scan, track, confidence (0-100%)
- **Temporal**: acq_date, acq_time, daynight
- **Target**: type (MODIS/VIIRS classification)

## 🚀 Week 1 Implementation

### ✅ Completed
- **Data Integration**: Combined 3-year dataset (2021-2023)
- **Data Quality**: 99%+ completeness, duplicate detection, validation  
- **Feature Engineering**: Temporal (season, month, hour) and geographic (regions)
- **EDA**: Distribution analysis, correlation studies, statistical tests
- **Visualization**: Multi-dimensional plots, heatmaps, trend analysis

### 📊 Key Findings
1. **Class Imbalance**: MODIS >> VIIRS observations
2. **Confidence Pattern**: Bimodal distribution (high/low confidence)
3. **Temporal Trends**: Seasonal and hourly fire detection patterns
4. **Geographic Patterns**: Regional variation across India
5. **Feature Correlations**: Strong thermal feature relationships

### 🛠️ Tech Stack  
**Core**: Python, Pandas, NumPy  
**Visualization**: Matplotlib, Seaborn  
**ML Ready**: Scikit-learn, XGBoost

## 🚀 Quick Start

```python
# Load and explore data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv('modis_2021_India.csv')
df2 = pd.read_csv('modis_2022_India.csv')  
df3 = pd.read_csv('modis_2023_India.csv')
df = pd.concat([df1, df2, df3], ignore_index=True)

# Basic exploration
print(f"Dataset shape: {df.shape}")
print(f"Fire types: {df['type'].value_counts()}")

# Visualize distributions
sns.countplot(data=df, x='type')
plt.show()
```

## 📈 Next Steps (Week 2+)
1. **Model Development**: Implement baseline classifiers
2. **Class Balancing**: Handle MODIS/VIIRS imbalance  
3. **Feature Selection**: Optimize feature set
4. **Model Evaluation**: Cross-validation and metrics
5. **Hyperparameter Tuning**: Optimize model performance


---
**Week 1 Status**: Data exploration and preprocessing ✅  
**Next Milestone**: Model development and training

## 🔗 References
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

**GitHub**: [arshdeepyadavofficial](https://github.com/arshdeepyadavofficial)  
**LinkedIn**: [Arshdeep Yadav](https://www.linkedin.com/in/arshdeep-yadav-827aa1257)

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

Made with ❤️ for environmental monitoring and disaster management

</div>



