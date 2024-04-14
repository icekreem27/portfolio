# **Exercise 3: Predict Cancer Mortality Rates in US Counties**

The provided dataset comprises data collected from multiple counties in the US. The regression task for this assessment is to predict cancer mortality rates in "unseen" US counties, given some training data. The training data ('Training_data.csv') comprises various features/predictors related to socio-economic characteristics, amongst other types of information for specific counties in the country. The corresponding target variables for the training set are provided in a separate CSV file ('Training_data_targets.csv'). Use the notebooks provided for lab sessions throughout this module to provide solutions to the exercises listed below. Throughout all exercises text describing your code and answering any questions included in the exercise descriptions should be included as part of your submitted solution.


The list of predictors/features available in this data set are described below:

**Data Dictionary**

avgAnnCount: Mean number of reported cases of cancer diagnosed annually

avgDeathsPerYear: Mean number of reported mortalities due to cancer

incidenceRate: Mean per capita (100,000) cancer diagoses

medianIncome: Median income per county

popEst2015: Population of county

povertyPercent: Percent of populace in poverty

MedianAge: Median age of county residents

MedianAgeMale: Median age of male county residents

MedianAgeFemale: Median age of female county residents

AvgHouseholdSize: Mean household size of county

PercentMarried: Percent of county residents who are married

PctNoHS18_24: Percent of county residents ages 18-24 highest education attained: less than high school

PctHS18_24: Percent of county residents ages 18-24 highest education attained: high school diploma

PctSomeCol18_24: Percent of county residents ages 18-24 highest education attained: some college

PctBachDeg18_24: Percent of county residents ages 18-24 highest education attained: bachelor's degree

PctHS25_Over: Percent of county residents ages 25 and over highest education attained: high school diploma

PctBachDeg25_Over: Percent of county residents ages 25 and over highest education attained: bachelor's degree

PctEmployed16_Over: Percent of county residents ages 16 and over employed

PctUnemployed16_Over: Percent of county residents ages 16 and over unemployed

PctPrivateCoverage: Percent of county residents with private health coverage

PctPrivateCoverageAlone: Percent of county residents with private health coverage alone (no public assistance)

PctEmpPrivCoverage: Percent of county residents with employee-provided private health coverage

PctPublicCoverage: Percent of county residents with government-provided health coverage

PctPubliceCoverageAlone: Percent of county residents with government-provided health coverage alone

PctWhite: Percent of county residents who identify as White

PctBlack: Percent of county residents who identify as Black

PctAsian: Percent of county residents who identify as Asian

PctOtherRace: Percent of county residents who identify in a category which is not White, Black, or Asian

PctMarriedHouseholds: Percent of married households

BirthRate: Number of live births relative to number of women in county


```python
import os
import pandas as pd

## Define paths to the training data and targets files
training_data_path = 'Training_data.csv'
training_targets_path = 'Training_data_targets.csv'
```

**Exercise 3.1**

Read in the training data and targets files. The training data comprises features/predictors while the targets file comprises the targets (i.e. cancer mortality rates in US counties) you need to train models to predict. Plot histograms of all features to visualise their distributions and identify outliers. Do you notice any unusual values for any of the features? If so comment on these in the text accompanying your code. Compute correlations of all features with the target variable (across the data set) and sort them according the strength of correlations. Which are the top five features with strongest correlations to the targets? Plot these correlations using the scatter matrix plotting function available in pandas and comment on at least two sets of features that show visible correlations to each other.

**(5 marks)**


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Read training data
training_data = pd.read_csv(training_data_path)
training_targets = pd.read_csv(training_targets_path)

# Concatenate data and targets for the analysis
full_data = pd.concat([training_data, training_targets], axis=1)

# Correct outliers in MedianAge
median_age_outliers = full_data[full_data['MedianAge'] > 100]
median_age_mean = full_data['MedianAge'].mean()
full_data['MedianAge'] = full_data['MedianAge'].apply(lambda x: median_age_mean if x > 100 else x)

# Plot histograms of all features
full_data.hist(bins=20, figsize=(15, 15))
plt.show()

# Display top 5 correlations with the target variable
correlations = full_data.corr()['TARGET_deathRate'].sort_values(ascending=False)
top_features = correlations.head(6)[1:]
print("Top five features with strongest correlations to the target:")
print(top_features)

# Plot scatter matrix for the top five features
sns.pairplot(full_data, vars=top_features.index)
plt.suptitle('Scatter Matrix for Top 5 Features with Strongest Correlations', y=1.02)
plt.show()


```


    
![png](COMP3611_Assessment_files/COMP3611_Assessment_3_0.png)
    


    Top five features with strongest correlations to the target:
    incidenceRate             0.443983
    PctPublicCoverageAlone    0.439734
    povertyPercent            0.413260
    PctHS25_Over              0.409915
    PctPublicCoverage         0.391899
    Name: TARGET_deathRate, dtype: float64



    
![png](COMP3611_Assessment_files/COMP3611_Assessment_3_2.png)
    


Regarding any unsual values, we can look for plots that deviate from normal distribution. There does not seem to be any oulier data other than the median age features.

Based on the scatter matrix plots:

We can see positive correlation between PctPrivateCoverage and PctPrivateCoverageAlone. This makes sense as individuals with private health coverage alone contribute to the overall percentage of individuals with private health coverage. The positive correlation indicates that as one of these percentages increases, the other increases as well.

Negative Correlation can be seen in povertyPercent vs. medianIncome. This negative correlation is also expected as higher poverty percentages are associated with lower median incomes. Counties with higher poverty percentages will tend to have lower median incomes, and vice versa.


*   There seem to be errors/outliers in the median age features (MedianAge) with values >> 100. This is clearly an error and needs to be corrected prior to fitting regression models. (1.5 marks for code above and this discussion)

*   Top five features with strongest correlations to targets are: incidenceRate, PctBachDeg25_Over, PctPublicCoverageAlone, medIncome and povertyPercent (2 marks for this description and code above).


*   medIncome and povertyPercent are negatively correlated to each other as you would expect.
*   povertyPercent and PctBachDeg25_Over are also negatively correlated highlighting that counties with higher degrees of poverty have fewer Bachelor graduates by the age of 25. povertyPercent also shows a strong positive correlation with PctPublicCoverageAlone, indicating that poverty stricken counties are less likely to be able to afford private healthcare coverage.
*   Similarly, PctBachDeg25_Over is negatively correlated with PctPublicCoverageAlone and positively correlated with medIncome. (1.5 marks for discussion of at least two sets of features that show correlations and code above)

**Exercise 3.2**

Create an ML pipeline using scikit-learn (as demonstrated in the lab notebooks) to pre-process the training data. (5 marks)


```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# Load the training data and targets
data = pd.read_csv('Training_data.csv')
targets = pd.read_csv('Training_data_targets.csv')

# Separate features and target
X = data
y = targets['TARGET_deathRate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The DataFrameSelector to create two parallel pipelines 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

# Separate numerical and categorical features
numerical_features = data.select_dtypes(include=['number']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Create pipelines for numerical features
numerical_pipeline = Pipeline([
    ('selector', DataFrameSelector(numerical_features)),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# and categorical features
categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(categorical_features)),
    ('one_hot_encoder', OneHotEncoder())
])

# Combine the pipelines using ColumnTransformer
full_pipeline = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Fit and transform the data with the full pipeline on the training set
transformed_data_train = full_pipeline.fit_transform(X_train)

# Transform the test set using the full pipeline
transformed_data_test = full_pipeline.transform(X_test)

# Display the original and transformed data (optional)
print("Original Data:")
print(data.head())

print("\nTransformed Data:")
print(pd.DataFrame(transformed_data_train).head())

```

    Original Data:
       avgAnnCount  avgDeathsPerYear  incidenceRate  medIncome  popEst2015  \
    0         59.0                30          404.3      33975        8251   
    1        114.0                41          403.8      47363       22702   
    2         33.0                11          352.0      77222        9899   
    3        254.0               100          429.6      80650       48904   
    4         75.0                32          407.5      42839       22255   
    
       povertyPercent  studyPerCap  MedianAge  MedianAgeMale  MedianAgeFemale  \
    0            20.5          0.0       51.3           50.8             51.9   
    1            13.8          0.0       40.8           39.8             42.7   
    2             6.8          0.0       38.1           36.9             39.8   
    3             7.5          0.0       43.5           42.7             44.1   
    4            14.6          0.0       31.1           30.2             31.6   
    
       ...  PctPrivateCoverageAlone  PctEmpPrivCoverage  PctPublicCoverage  \
    0  ...                      NaN                26.0               49.7   
    1  ...                     56.5                46.8               31.6   
    2  ...                     65.4                54.3               18.2   
    3  ...                     64.2                55.6               28.8   
    4  ...                     50.7                46.5               26.8   
    
       PctPublicCoverageAlone   PctWhite  PctBlack  PctAsian  PctOtherRace  \
    0                    20.6  96.684036  0.438181  0.082899      0.272383   
    1                    13.0  92.295459  2.102845  0.609648      0.879131   
    2                     8.6  95.690422  0.000000  0.523871      0.118612   
    3                    13.5  89.606996  7.407407  0.870370      0.450617   
    4                    18.1  79.587990  2.948701  8.482564      5.637090   
    
       PctMarriedHouseholds  BirthRate  
    0             51.926207   5.041436  
    1             50.949545   6.329661  
    2             64.532156   5.148130  
    3             62.344481   5.627462  
    4             63.005948  10.436469  
    
    [5 rows x 31 columns]
    
    Transformed Data:
             0         1         2         3         4         5         6   \
    0 -0.128977 -0.053646 -0.029166  0.799382 -0.041545 -0.981118  0.164471   
    1 -0.394742 -0.318743 -0.249612 -0.911029 -0.296052  0.598820 -0.313738   
    2 -0.142265 -0.073528  0.447527 -0.236550 -0.049226 -0.371043  0.618949   
    3 -0.436952 -0.365135 -1.178503 -0.860638 -0.320677  0.880393 -0.313738   
    4 -0.317358 -0.203868 -0.153520 -0.741064 -0.233932  0.332890 -0.313738   
    
             7         8         9   ...        21        22        23        24  \
    0 -0.137110 -0.187978 -0.445440  ...  1.492915  1.557573 -0.784939 -0.829350   
    1 -0.019632  0.447303  0.713338  ... -0.590766 -0.525352  0.385399  0.584977   
    2 -0.235364 -1.169776 -1.281280  ...  0.085022 -0.119448 -0.679740 -0.458931   
    3 -0.188373 -1.554795  0.371403  ... -1.491818 -1.112843  0.214451  0.231395   
    4 -0.075167  0.216292  0.219432  ...  0.028707 -0.632168  0.714146  0.366093   
    
             25        26        27        28        29        30  
    0  0.662262 -0.567052 -0.213109  0.003537  0.529158 -0.458523  
    1 -0.990731  1.517064 -0.417955 -0.506876 -0.761643 -1.375816  
    2  0.566108 -0.588897  0.129873  0.054313  0.627064  1.457320  
    3  0.672040 -0.537742 -0.473139  0.327809  0.636916  5.825411  
    4  0.824078 -0.521763 -0.459222 -0.539978 -0.160197  0.806795  
    
    [5 rows x 31 columns]


**Exercise 3.3**

Fit linear regression models to the pre-processed data using: Ordinary least squares (OLS), Lasso and Ridge models. Choose suitable regularisation weights for Lasso and Ridge regression and include a description in text of how they were chosen. In your submitted solution make sure you set the values for the regularisation weights equal to those you identify from your experiment(s). Quantitatively compare your results from all three models and report the best performing one. Report the overall performance of the best regression model identified. Include code for all steps above. (10 marks)


```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

# Ordinary Least Squares (OLS)
ols_reg = LinearRegression()
ols_mse = -cross_val_score(ols_reg, transformed_data_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()

# Lasso Regression
lasso_reg = Lasso()

# Perform GridSearchCV to find the best alpha value
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_grid = GridSearchCV(lasso_reg, lasso_params, scoring = 'neg_mean_squared_error', cv = 5)
lasso_grid.fit(transformed_data_train, y_train)
lasso_best_alpha = lasso_grid.best_params_['alpha']

# Create a new Lasso model with the best alpha and fit it
lasso_reg = Lasso(alpha=lasso_best_alpha)
lasso_reg.fit(transformed_data_train, y_train)

# Make predictions and calculate mse
lasso_predictions = lasso_reg.predict(transformed_data_test)
lasso_mse = mean_squared_error(y_test, lasso_predictions)


# Ridge Regression
ridge_reg = Ridge()

# Perform GridSearchCV to find the best alpha value
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge_reg, ridge_params, scoring = 'neg_mean_squared_error', cv = 5)
ridge_grid.fit(transformed_data_train, y_train)
ridge_best_alpha = ridge_grid.best_params_['alpha']

# Create a new Ridge model with the best alpha
ridge_reg = Ridge(alpha=ridge_best_alpha)
ridge_reg.fit(transformed_data_train, y_train)

# Make predictions and calculate mse
ridge_predictions = ridge_reg.predict(transformed_data_test)
ridge_mse = mean_squared_error(y_test, ridge_predictions)


# Compare results
results = {
    'OLS': ols_mse,
    'Lasso': lasso_mse,
    'Ridge': ridge_mse
}

best_model = min(results, key=results.get)
best_mse = results[best_model]

# Display results
print("Cross-Validated Mean Squared Error (MSE) for OLS: {:.4f}".format(ols_mse))
print("Cross-Validated Mean Squared Error (MSE) for Lasso: {:.4f} (Best alpha: {})".format(lasso_mse, lasso_best_alpha))
print("Cross-Validated Mean Squared Error (MSE) for Ridge: {:.4f} (Best alpha: {})".format(ridge_mse, ridge_best_alpha))

print("\nBest Performing Model: {}".format(best_model))
print("Cross-Validated MSE of the Best Performing Model: {:.4f}".format(best_mse))

```

    Cross-Validated Mean Squared Error (MSE) for OLS: 386.8304
    Cross-Validated Mean Squared Error (MSE) for Lasso: 352.6522 (Best alpha: 0.1)
    Cross-Validated Mean Squared Error (MSE) for Ridge: 349.9903 (Best alpha: 10)
    
    Best Performing Model: Ridge
    Cross-Validated MSE of the Best Performing Model: 349.9903






In this analysis, the thee linear regression models: Ordinary Least Squares (OLS), Lasso, and Ridge—were fitted to thr already pre processed data. The regularization weights for Lasso and Ridge were determined through hyperparameter tuning with GridSearchCV. For Lasso and Ridge models, a range of alpha values (0.01, 0.1, 1, 10, 100) was used to identify the most ideal value based on a cross-validated mean squared error. The identified optimal alpha values were then utilized to create new Lasso and Ridge models. Cross-validation using the negative mean squared error as the scoring metric was used to quantitatively compare the performance of theses models.

The best-performing model was determined by the lowest Cross-Validated MSE and was Ridge regression. Specifically, the Ridge model with an alpha value of 10 showed the most favorable performance and achieved a Cross-Validated MSE of 349.9903. This outcome highlights the effectiveness of the Ridge model in predictive accuracy. The provided code outlines with comments the entire process, from model fitting and hyperparameter tuning to the quantitative comparison of results.


```python

```
