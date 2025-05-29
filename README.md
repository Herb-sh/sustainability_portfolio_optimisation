# Financial Data Insights: AI-Driven Investment Optimization for Sustainability

This repository contains code, data processing steps, and evaluation scripts developed as part of 
the master's thesis:

**"Financial Data Insights: Assessing the Efficacy of AI-Driven Investment Optimization Strategies
for Sustainability. A Machine Learning Approach"**

The goal of this work is to explore the use of machine learning techniques in portfolio 
optimization, while placing particular emphasis on sustainability through ESG-compliant 
company selection.

---

## Project Summary

Portfolio optimization is often approached either through:

- **Return-oriented strategies**, which focus on maximizing returns or reducing volatility, and
- **Technique-driven approaches**, which emphasize the evaluation of machine learning models, 
sometimes without a direct focus on returns.

This project attempts to consider both perspectives. It uses a custom dataset, manually prepared 
from publicly available sources, and evaluates three models across different configurations.

---

## Key Aspects

- **Sustainability-focused selection**: Only companies with above-average ESG ratings were included.
- **Custom dataset**: Data was manually gathered and cleaned to match the requirements of the project.
- **Models evaluated**:
    - Long Short-Term Memory (LSTM)
    - Prophet
    - XGBoost
- **Benchmark**: Traditional Modern Portfolio Theory (Mean-Variance Optimization) was used as a baseline.
- **Multivariate modeling**: Static features were incorporated alongside time-series data to improve training.
- **Multiple time windows**: Models were evaluated using 1-month, 6-month, and 12-month prediction horizons.

---


 # Project documentation

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sustainable-portfolio-optimization.git
   cd sustainable-portfolio-optimization
   ```

2. Install the required libraries:
```bash
    pip install -r requirements.txt
```


