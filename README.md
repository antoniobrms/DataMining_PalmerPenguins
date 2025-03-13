# Palmer Penguins Data Mining Project

## Overview
This project analyzes the Palmer Penguins dataset, which includes measurements for three penguin species found on three islands in the Palmer Archipelago, Antarctica. The goal is to perform data cleaning, exploratory data analysis (EDA), feature engineering, and apply a logistic regression model to classify the penguin species.

## Dataset
The dataset used contains measurements of:
- **Culmen Length** (in mm)
- **Culmen Depth** (in mm)
- **Flipper Length** (in mm)
- **Body Mass** (in grams)
- **Penguin Species** (Adelie, Gentoo, Chinstrap)

## Methodology

### 1. Data Cleaning
- Missing values in numerical columns were filled with the **mean**.
- Missing values in categorical columns were filled with the **mode**.

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions using **histograms** and **box plots**.
- Analyzed relationships between variables using **pair plots**.
- Key findings:
  - **Culmen Length** ranges between 35-50 mm.
  - **Flipper Length** ranges between 170-230 mm.
  - Positive correlations between flipper length, culmen length, and body mass.

### 3. Feature Engineering
- Added a new feature by multiplying **culmen length** and **culmen depth**.
- Categorized **body mass** into three classes: *light*, *medium*, and *heavy*.

### 4. Model Building
- Used **logistic regression** for species classification.
- Data was split into training and testing sets.
- Features were standardized using `StandardScaler` to optimize performance.

### 5. Model Evaluation
- Achieved **98.5% accuracy** on the test set.
- Evaluated performance using precision, recall, and F1-score.

## Key Insights
- **Flipper length** is a key predictor of species classification.
- **Culmen measurements** (length and depth) also help distinguish species.
- Categorizing **body mass** improved model performance.

## Conclusion
The logistic regression model performed well with high accuracy in classifying penguin species. Feature engineering and EDA contributed to improved predictive power. Future improvements could focus on handling missing data more effectively and increasing the dataset size for better generalization.

## References
- [Seaborn Documentation for Data Visualization](https://seaborn.pydata.org/)

## Installation
To run the analysis, you need to install the required libraries:
```bash
pip install numpy pandas seaborn scikit-learn matplotlib
```
## Usage
1. **Prepare the dataset**: Ensure the files `penguins_size.csv` and `penguins_iter.csv` are in the same directory as the script.
2. **Run the main script**: Execute the `PalmerPenguinDataset.py` script to clean the data, perform exploratory data analysis, and train the logistic regression model:
   ```bash
   python PalmerPenguinDataset.py







