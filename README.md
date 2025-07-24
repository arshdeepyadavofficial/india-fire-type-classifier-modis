# 🔥 MODIS Fire Type Classification for India (2021-2023)

**Machine Learning Classificatio### 📊 Advanced Key Findings of Fire Types Using eractive Fire Maps**: Folium-based geographic visualization with color-coded fire typesA MODIS Satellite Data**

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
- **Basic Visualization**: Multi-dimensional plots, heatmaps, trend analysis

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

## 🎨 Week 2 Implementation

### ✅ Enhanced Visualizations & Analytics
- **🗺️ Interactive Fire Maps**: Folium-based geographic visualization with color-coded fire types
- **📊 Enhanced Correlation Analysis**: Improved heatmaps with custom color schemes and better readability  
- **🎯 Advanced Distribution Plots**: Multi-panel visualizations for comprehensive data exploration
- **⏰ Time Series Analysis**: Temporal patterns with trend analysis and seasonal insights
- **🌍 Geographic Density Maps**: Hexbin plots and scatter visualizations for spatial analysis
- **🚀 Feature Importance Visualization**: Enhanced bar charts for model interpretability
- **💫 Professional Styling**: Custom color palettes, animations, and enhanced plot aesthetics

### 🛠️ Extended Tech Stack
**Core**: Python, Pandas, NumPy  
**Visualization**: Matplotlib, Seaborn, Folium (Interactive Maps)  
**ML Ready**: Scikit-learn, XGBoost  
**Statistical**: SciPy, Feature Engineering Tools

### � Advanced Key Findings
1. **Interactive Geographic Patterns**: Regional fire distribution mapped with clustering analysis
2. **Enhanced Correlation Insights**: Strong thermal feature relationships with custom visualizations
3. **Temporal Deep Dive**: Comprehensive time series analysis with seasonal patterns
4. **Spatial Distribution**: Geographic clustering and density mapping insights
5. **Performance Optimization**: Efficient rendering for large datasets (up to 10K+ points)

### 🎨 Interactive Features
- **Clickable Maps**: Detailed fire information popups with enhanced styling
- **Multi-Theme Support**: Dark themes, satellite views, and street maps
- **Real-time Analytics**: Dynamic visualization updates and filtering
- **Professional Output**: Publication-ready plots with custom branding

### 💻 Week 2 Quick Start - Interactive Maps

```python
import folium
from folium import plugins

# Interactive map visualization
sample_points = df.sample(n=5000, random_state=42)
fire_map = folium.Map(location=[20.5937, 78.9629], zoom_start=6)

# Add enhanced markers with popups
for idx, row in sample_points.iterrows():
    color = 'red' if row['type'] == 2 else 'orange' if row['type'] == 1 else 'green'
    fire_type = 'High Confidence' if row['type'] == 2 else 'Moderate' if row['type'] == 1 else 'Vegetation'
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4, fillColor=color, color='black', weight=1, fillOpacity=0.8,
        popup=f"""
        <b>Fire Type:</b> {fire_type}<br>
        <b>Confidence:</b> {row['confidence']}%<br>
        <b>Date:</b> {row['acq_date']}<br>
        <b>Coordinates:</b> {row['latitude']:.3f}, {row['longitude']:.3f}
        """
    ).add_to(fire_map)

# Add heatmap layer
heat_data = [[row['latitude'], row['longitude']] for idx, row in sample_points.iterrows()]
plugins.HeatMap(heat_data).add_to(fire_map)

fire_map
```


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



