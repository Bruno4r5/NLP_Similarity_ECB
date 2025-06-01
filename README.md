# The Similarity of ECB's Communication: Replication & Extension

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Academic Project](https://img.shields.io/badge/Type-Academic%20Research-orange.svg)](https://github.com)

## 📋 Project Overview

This project replicates and extends the paper **"The Similarity of ECB's Communication"** by Diego Amaya and Jean-Yves Filbien (2015). As part of our Master's program, we analyzed European Central Bank (ECB) communication patterns from 1999 to 2024, examining how similarity in monetary policy statements affects market reactions.

### 🎯 Key Objectives

- **Replicate** the original study's methodology and findings
- **Extend** the analysis from 2013 to October 2024 (adding 11 years of data)
- **Enhance** the research with new sentiment analysis techniques
- **Investigate** the impact of major economic events (COVID-19 crisis) on communication patterns

## 👥 Team Members

- **Bruno Sciascia**
- **Thomas Grangaud** 
- **Titouan Guesdon**
- **Nicolas Blache**


## 🔬 Research Methodology

### Data Collection & Processing
1. **Web Scraping**: Automated extraction of ECB press conference transcripts using Selenium
2. **Data Cleaning**: Removal of Q&A sections, standardization of dates, duplicate handling
3. **Tokenization**: NLP preprocessing with NLTK (stopword removal, Porter stemming)

### Key Metrics Computed
- **Pessimism Score**: Sentiment analysis using Loughran-McDonald financial dictionary
- **Jaccard Similarity**: Bigram-based similarity between consecutive statements
- **Cumulative Abnormal Returns (CAR)**: Market reaction measurement using STOXX 50 index

### Novel Extensions
- **Subjectivity Analysis**: Using TextBlob for objectivity measurement
- **Advanced NLP**: DistilBERT transformer model for financial sentiment classification
- **Extended Timeline**: Analysis through COVID-19 and recent monetary policy changes

## 📊 Key Findings

### Original Paper Replication
- ✅ **Confirmed**: ECB communication similarity increases over time
- ✅ **Validated**: Inflation negatively impacts communication similarity
- ⚠️ **Modified**: Lower trend coefficients due to COVID-19 disruption

### New Insights (2014-2024)
- **COVID-19 Impact**: Significant drop in similarity during 2020-2021
- **Market Predictability**: Reduced effectiveness of sentiment-based market prediction post-crisis
- **Communication Evolution**: Recovery in similarity patterns post-2022

## 🛠️ Technical Implementation

### Dependencies
```python
# Core Libraries
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0

# Web Scraping
selenium>=4.0.0
beautifulsoup4>=4.9.0
requests>=2.25.0

# NLP & Sentiment Analysis
nltk>=3.6.0
textblob>=0.15.0
transformers>=4.5.0

# Statistical Analysis
statsmodels>=0.12.0
scikit-learn>=0.24.0

# Data Processing
openpyxl>=3.0.0
```

### Project Structure
```
ecb-communication-analysis/
│
├── Part1_Web_Scrapping_ECB.py          # Data collection script
├── Part2_Computation.py                # Main analysis pipeline
├── data/
│   ├── database.xlsx                  # Scraped ECB transcripts
│   ├── mro_data.csv                   # Main Refinancing Operations data
│   ├── gdp_data.csv                   # GDP time series
│   ├── hcpi_data.csv                  # Inflation data
│   ├── SX5E_hist_price.csv            # STOXX 50 historical prices
│   └── extension.csv                  # Subjectivity scores
├── results/
│   └── stats_table.pdf                # Descriptive statistics
├── report_replication_The Similarity of ECBs Communication.pdf
└── README.md
```

### Academic Contributions
1. **Temporal Validation**: Confirmed robustness of original findings over 25-year period
2. **Crisis Analysis**: Documented communication pattern changes during economic uncertainty
3. **Methodological Enhancement**: Integrated modern NLP techniques with traditional finance metrics

## 🤝 Contributing

This is an academic project completed as part of our Master's program. For questions or discussions about our methodology, please open an issue or contact the team members.

---

*This project demonstrates the application of modern data science techniques to financial communication analysis, bridging traditional econometric methods with contemporary NLP approaches.*
