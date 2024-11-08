# Sagemaker Work
## AWS Sagemaker Deliverables for

## Project 1 - 09/25/24 Detailed Deliverables (Focus on Future MLOps)

### Deliverable: A template notebook (with detailed instructions) that performs the following for a public dataset & use case:

* USED DATASET FOR DELIVERABLE
    * https://www.kaggle.com/code/gauravduttakiit/bike-sharing-multiple-linear-regression/notebook


* Processes raw data into features (cleaning, formatting, validating, binning, etc). Make sure it includes some binning
    * DATA QUALITY checks include:
        * NULL/missing values
        * Duplicate values
        * Removal of redundant/unnecessary columns 
    * Conversion types (especially from meaningless integer to categorical values)
    * Binning refers to “transforming continuous data into intervals/bins”
        * Dummy variables to prepare data for ML model
        * Technique to drop first variable from each set of dummy variables to prevent multicollinearity


* Registers features into a new feature group in Sagemaker Feature Store
    * Documentation Link: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html
    * Created a script to automate this process
    * Initialize Environment:
        * Import necessary libraries (boto3, pandas, sagemaker, etc.)
        * Initialize a SageMaker session and Boto3 clients for SageMaker Feature Store
    * Load and Prepare Data:
        * Load the dataset into a DataFrame 
        * Add columns back to the DataFrame for feature store registration (if needed for date time stamps and key identifiers)
        * Handle missing values by filling them with placeholders
    * Define Feature Group:
        * Define the feature group name 
        * Specify feature definitions, including all relevant columns 
    * Create Feature Group:
        * Check if the feature group already exists and delete it if it does
        * Create the feature group in SageMaker Feature Store with the specified schema
    * Wait for Feature Group Activation:
        * Continuously check the status of the feature group until it becomes ACTIVE
        * Will be created on AWS Sagemaker Feature Store before it programmatically says it has been active (check the store periodically as script runs)
    * Ingest Data:
        * Ingest the prepared DataFrame (bike_new) into the feature group once it is active
    * Completion Message:
        * Print a success message indicating that the features have been registered into the feature group successfully


* Train & test a linear regression model to predict some outcome
    * Split data into train and test using sklearn model
        * Usual: train is around 70% and test is around 30%
    * Scale all numerical variables using MinMaxScaler() function built into sklearn
        * Scales all numerical features within a given range 
    * Apply the scale to the numerical variables (fit_transform function)
    * Model creation in steps
        * Split the training data into X and Y training data
    * Calculate VIFs (variance inflation factors) - just shows multicollinearity chances within the model 
    * Create the first-fitted OLS (ordinary least squares) model -  starting point for all spatial regression analyses
        * Check the important parameters obtained from the model
    * Print the final summary of the regression model using those variables 
    * Results of bike dataset (and how results for regression model will look):
        * Model Summary:
            * Dependent Variable: cnt
            * R-squared: 1.000
            * Adjusted R-squared: 1.000
            * F-statistic: 7.327e+31
            * Prob (F-statistic): 0.00
            * Number of Observations: 510
            * AIC (Akaike Information Criterion): -2.643e+04
            * BIC (Bayesian Information Criterion): -2.636e+04
            * Df Residuals: 494
            * Df Model: 15
            * Covariance Type: nonrobust
        * Coefficients:
            * const: -1.805e-12 (p-value: 0.000)
            * casual: 1.0000 (p-value: 0.000)
            * registered: 1.0000 (p-value: 0.000)
            * mnth_1: -1.563e-12 (p-value: 0.000)
            * mnth_2: -4.636e-13 (p-value: 0.243)
            * mnth_3: -7.252e-13 (p-value: 0.022)
            * mnth_4: -1.084e-12 (p-value: 0.000)
            * mnth_6: 4.539e-13 (p-value: 0.064)
            * mnth_9: 1.794e-13 (p-value: 0.438)
            * mnth_10: -5.88e-13 (p-value: 0.032)
            * mnth_12: -5.347e-13 (p-value: 0.068)
            * weekday_0: 1.599e-12 (p-value: 0.000)
            * weekday_6: -5.365e-13 (p-value: 0.020)
            * season_1: 3.313e-13 (p-value: 0.326)
            * season_4: 1.004e-13 (p-value: 0.621)
            * weathersit_1: 5.898e-13 (p-value: 0.000)
        * Diagnostic Tests:
            * Omnibus: 5.990 (p-value: 0.050)
            * Durbin-Watson: 1.823
            * Jarque-Bera (JB): 6.049 (p-value: 0.0486)
            * Skew: -0.266
            * Kurtosis: 2.960
            * Condition Number: 4.63e+04
        * Interpretation:
            * R-squared and Adjusted R-squared:
                * R-squared (1.000): Indicates that 100% of the variability in the dependent variable (cnt) is explained by the independent variables in the model. This is an unusually high value, suggesting a perfect fit, which might indicate overfitting.
                * Adjusted R-squared (1.000): Adjusts the R-squared value for the number of predictors in the model. It is also 1.000, reinforcing the perfect fit.
            * F-statistic and Prob (F-statistic):
                * F-statistic (7.327e+31): Tests the overall significance of the model. A very high F-statistic value indicates that the model is statistically significant.
                * Prob (F-statistic) (0.00): The p-value associated with the F-statistic is zero, indicating that the model is statistically significant.
            * Coefficients and P-values:
                * Coefficients: Represent the change in the dependent variable for a one-unit change in the independent variable, holding all other variables constant.
                * P-values: Indicate the statistical significance of each coefficient. A p-value less than 0.05 typically indicates that the coefficient is statistically significant.
                * Significant Variables: casual, registered, mnth_1, mnth_3, mnth_4, mnth_10, weekday_0, weekday_6, weathersit_1.
            * Diagnostic Tests:
                * Durbin-Watson (1.823): Tests for autocorrelation in the residuals. A value close to 2 indicates no autocorrelation.
                * Omnibus and Jarque-Bera (JB) Tests: Test for normality of the residuals. Significant p-values indicate that the residuals are not normally distributed.
                * Skew and Kurtosis: Measure the asymmetry and peakedness of the residual distribution. Values close to zero indicate normality.
                * Condition Number (4.63e+04): Indicates potential multicollinearity. A high condition number suggests that some predictors may be highly correlated.
            * Conclusion:
                * Perfect Fit: The R-squared and Adjusted R-squared values of 1.000 indicate a perfect fit to the training data. This is unusual and may suggest overfitting.
                * Statistical Significance: The overall model and most of the individual predictors are statistically significant.
                * Potential Issues: The diagnostic tests suggest potential issues with normality of residuals and multicollinearity among predictors.


* Register the model in Sagemaker’s Model Registry
    * Lauren’s red wine dataset has not implemented this yet
    * Documentation Link: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html
    * Allows for 
        * Catalog models for production
        * Manage model versions
        * Associate metadata, such as training metrics, with a model
    * Created script to automate this process
    * This script sets up a SageMaker session and Boto3 client, retrieves the necessary image URI for the scikit-learn framework, and generates a unique model package group name using the current timestamp to avoid conflicts
    * Then creates the model package group and waits for its successful creation
    * Defines the model using the retrieved image URI and model data, and prepares the inference specification for the model package
    * Registers the model package with the specified group name, description, and approval status, and finally, it updates the model package status to “Approved”


* Create a Sagemaker Pipeline to trigger when new raw data appears in S3 to process the data, load into feature, run thru model to generate predictions, and write predictions back to a file in S3
    * Lauren’s red wine dataset has not implemented this yet
    * Past Knowledge
        * Similar to what was worked on during the summer
        * Create programmatically, not through the Sagemaker UI
    * Documentation Links
        * https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html
        * https://docs.aws.amazon.com/sagemaker/latest/dg/run-pipeline.html
        * https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-feature-processor-schedule-pipeline.html
     



## Project 2 - 10/15/24 Model Registry, GitHub Connection, Model Cards, Pickle File

### Deliverable 1: register model in model registry with model card (metadata about the model - tailored to development team + 5/3 information standard)

* Model Card Documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/model-cards.html
* Required steps for model metadata (5/3 standard):
    * 
* Optional steps for model metadata (team-specific use case):
    * 


### Deliverable 2: establish live connection to remote GitHub repository to pull changes when account deactivation occurs

* GitHub Connection Documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-resource.html
* Steps for connection:
    * 


### Deliverable 3 (Stretch Goal): export trained weighted coefficients into .pickle file (in 1 notebook), then pass in new data through the .pickle file to create more predictions (separate notebook)

* Create trained models (for bike share predictions data, it is called lr1)
    * Creation step
        * Created by taking the splits of data (X_split, Y_split) and then running fit(), sees the coefficients generated against the training data
        * Run model name .params - lr1.params - and then see all the important features from RFE (removed least significant features recursively) + respective coefficients
        * Export this into .pickle file (did .joblib in the bike share predictions data) - make change for that 
    * Run step 
        * Upload .pickle file into a new notebook
        * Using coefficients from the file, run new dataset through it
        * Compare predictions from model through pickle file, with original indicators from dataset
    * Documentation links 
        * https://stackoverflow.com/questions/68214823/how-to-deploy-machine-learning-model-saved-as-pickle-file-on-aws-sagemaker
        * https://www.analyticsvidhya.com/blog/2021/08/quick-hacks-to-save-machine-learning-model-using-pickle-and-joblib/

