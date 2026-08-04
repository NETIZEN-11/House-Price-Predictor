# 🏠 House Price Predictor

A modern, AI-powered web application for predicting house prices using machine learning. Built with Flask and featuring a beautiful, responsive UI.

![House Price Predictor](https://img.shields.io/badge/Status-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.0+-red) ![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## ✨ Features

### 🎨 **Modern UI/UX Design**
- **Gradient Background**: Beautiful purple-to-blue gradient design
- **Glass-morphism Effect**: Modern transparent containers with blur effects
- **Responsive Layout**: Perfect on desktop, tablet, and mobile devices
- **Interactive Elements**: Smooth animations and hover effects
- **Professional Typography**: Clean, readable Inter & Poppins fonts

### 🧠 **Machine Learning**
- **Intelligent Predictions**: Uses trained Random Forest model
- **13 Feature Analysis**: Comprehensive property evaluation
- **Real-time Processing**: Instant predictions with loading animations
- **Accurate Results**: Properly scaled and formatted output

### 🔧 **User Experience**
- **Icon-based Labels**: Clear, descriptive input fields
- **Tooltips**: Detailed explanations for each feature
- **Sample Data**: One-click sample data filling for testing
- **Input Validation**: Real-time form validation
- **Error Handling**: Graceful error management

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/NETIZEN-11/House-Price-Predictor.git
cd House-Price-Predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open your browser**
Navigate to `http://127.0.0.1:5000`

## 📊 Dataset Features

The model uses 13 key features to predict house prices:

| Feature | Description | Example |
|---------|-------------|---------|
| **CRIM** | Per capita crime rate by town | 0.00632 |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft. | 18.0 |
| **INDUS** | Proportion of non-retail business acres per town | 2.31 |
| **CHAS** | Charles River dummy variable (1 if bounds river; 0 otherwise) | 0 |
| **NOX** | Nitric oxides concentration (parts per 10 million) | 0.538 |
| **RM** | Average number of rooms per dwelling | 6.575 |
| **AGE** | Proportion of owner-occupied units built prior to 1940 | 65.2 |
| **DIS** | Weighted distances to five Boston employment centres | 4.0900 |
| **RAD** | Index of accessibility to radial highways | 1 |
| **TAX** | Full-value property-tax rate per $10,000 | 296 |
| **PTRATIO** | Pupil-teacher ratio by town | 15.3 |
| **B** | Population index by town | 396.90 |
| **LSTAT** | % lower status of the population | 4.98 |

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, Random Forest
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with modern design principles
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Inter, Poppins)

## 📱 API Endpoints

### Web Interface
- `GET /` - Main prediction interface
- `POST /predict` - Submit prediction form

### API
- `POST /predict_api` - JSON API for predictions

#### Example API Usage
```bash
curl -X POST http://127.0.0.1:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "CRIM": 0.00632,
      "ZN": 18.0,
      "INDUS": 2.31,
      "CHAS": 0,
      "NOX": 0.538,
      "RM": 6.575,
      "Age": 65.2,
      "DIS": 4.0900,
      "RAD": 1,
      "TAX": 296,
      "PTRATIO": 15.3,
      "B": 396.90,
      "LSTAT": 4.98
    }
  }'
```

## 🎯 Model Performance

- **Algorithm**: Random Forest Regressor
- **Features**: 13 housing characteristics
- **Preprocessing**: StandardScaler normalization
- **Output**: Price prediction in USD format

## 🖥️ Screenshots

### Before vs After UI Comparison
The application features a complete UI overhaul from a basic form to a modern, professional interface:

- ✅ Modern gradient backgrounds
- ✅ Glass-morphism design elements  
- ✅ Responsive grid layout
- ✅ Interactive hover effects
- ✅ Loading animations
- ✅ Professional typography

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
The application is ready for deployment on platforms like:
- Heroku
- Vercel
- Railway
- PythonAnywhere

### Environment Variables
Set up the following for production:
```
FLASK_ENV=production
PORT=5000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original dataset inspiration from Boston Housing dataset
- UI/UX design inspired by modern web applications
- Icons provided by Font Awesome
- Fonts by Google Fonts

## 📞 Contact

**NETIZEN-11** - [GitHub Profile](https://github.com/NETIZEN-11)

Project Link: [https://github.com/NETIZEN-11/House-Price-Predictor](https://github.com/NETIZEN-11/House-Price-Predictor)

---

⭐ **Star this repository if you found it helpful!** ⭐

