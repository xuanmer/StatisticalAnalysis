# StatisticalAnalysis

A Python toolkit for statistical analysis, integrating common statistical methods and visualization functionalities to help you efficiently process data, perform exploratory analysis, and present results.

## Features

- Core statistical methods: hypothesis testing, ANOVA, correlation analysis, regression, etc.
- Convenient data loading and preprocessing utilities
- Built-in visualizations: histograms, box plots, scatter plots, etc.
- Modular, easy to extend and customize
- Example scripts and clear documentation

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scipy

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Repository Structure
```bash
StatisticalAnalysis/
│
├── analysis/                # Core analysis modules
│   ├── anova.py             # Functions for ANOVA
│   ├── correlation.py       # Correlation analysis functions
│   ├── regression.py        # Regression analysis functions
│   ├── hypothesis_test.py   # Hypothesis testing utilities
│   └── ...
│
├── utils/                   # Helper utilities
│   └── data_loader.py       # CSV/Excel/JSON data loading helpers
│
├── data/                    # Sample datasets
│   └── sample.csv
│
├── examples/                # Usage examples
│   └── example_analysis.py
│
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

---

## Usage Examples
### Correlation Analysis
```python
from analysis.correlation import correlation_analysis
import pandas as pd

# Load sample data
df = pd.read_csv('data/sample.csv')

# Compute Pearson and Spearman correlation between two columns
result = correlation_analysis(df, 'column1', 'column2')
print(result)
```

### ANOVA
```python
from analysis.anova import one_way_anova
import pandas as pd

df = pd.read_csv('data/sample.csv')
anova_table = one_way_anova(df, 'group_column', 'value_column')
print(anova_table)
```
Check the examples/ folder for more complete scripts covering regression, hypothesis tests, and plotting.

---

## License
This project is licensed under the MIT License.