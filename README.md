# Credit Card Default Prediction

The project is about predicting whether a particular user will go into default for credit card payments given certain features. I will be using the Credit Card Default dataset from the UCI Machine Learning Repository. The dataset can be found here - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#

I will model the problem as a classification problem, with the output variable being 0 or 1.

## Project Set Up and Installation
The project was set up using the following resources - 
Compute Instance - STANDARD_DS2_V2
Compute Cluster - STANDARD_DS2_V2

## Dataset

### Overview
I have downloaded the data from the link mentioned above. I create a Dataset in the AzureML Studio, so I can readily access the raw dataset whenever I need it. The dataset contains 30,000 rows and 24 different attributes. There are no missing values so the dataset is clean. The features are explained below - 

X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 
X2: Gender (1 = male; 2 = female). 
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year). 
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. 
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005. 

### Task
The task of this project is to use Machine Learning in the Risk Management domain. To solve the classification problem, various techniques can be used such as RandomForest, LogisticRegression, XGBoost etc.

### Access
The dataset can be accessed using the Datasets tab in the AzureML Studio
![Dataset](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/dataset.png)

## Automated ML
To use the AzureML AutoML feature, I decide to run it for 20 minutes, since the dataset is small and a model trains quite quickly. Since I am not using a validation dataset, it is important to use n-fold validation for which I choose n=4. I also use the compute cluster that I initialize, and use Accuracy as the primary metric. We also need to tell it about which column to use as Y. After submitting an AutoML run, I use the RunDetails widget to track the progress. 

![Config](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/automl-config.png)

![RunDetails](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/autoML-run-widget.png)

### Results

After letting the AutoML run, we see a variety of different algorithms that it has chosen. The best model chosen is the Voting Ensemble Method, which uses a bunch of other models and cleverly uses their predictions to create a final prediction. I could have improved AutoML by letting it run for more time since that would have given it more opportunity to try new models. Another way of improvement is to use a different Primary Metric such as AUC or F1-Score.

![BestModel](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/autoML-best-model.png)

The best model has the following metrics and parameters - 

![BestModel_metrics](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/automl-best-model-metrics.png)
![BestModel_params](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/automl-best-model-parameters.png)


## Hyperparameter Tuning

For the Hyperparameter Tuning part, I have decided to use a Logistic Regression Model. The simplicity of the model means that the hyperparameters can be changed with good effect. I have deicded to use 2 hyperparameters to optimize - C which is the regularization parameter, and l1_ratio parameter, which decides the weighting between L1 and L2 regularization. I am using the 'elasticnet' training method to train the model.

![hyper_params](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/hyperdrive-params-optimize.png)

C is varied on a logarithmic scale, while the l1_ratio has to be a value between 0 and 1 and is varied accordingly. 

I use a BanditPolicy as early termination since I don't want experiments where Accuracy is less than 0.1 of the best. The config settings just include the training script, the environment, the primary metric which I have decided to be accuracy since we are solving a classification problem.

We can track the progress of the run through the RunDetails widget. 

![hyper_widget1](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/hyperdrive-widget-1.png)
![hyper_widget2](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/hyperdrive-widget-2.png)

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

After running Hyperdrive, we see a variety of runs with different hyperparameters. The best model is shown below with its parameters - 

![hyper_best_run](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/best-run-hyperdrive.png)

As we can see, the best accuracy is obtained when C=0.01 & l1=0.9. I could have improved the results by using a different algorithm, or by trying different normalizations before training the model. It would also have been beneficial to search a wider space of hyper-parameters.


## Model Deployment
Since the model trained by AutoML is much better in accuracy, I decide to deploy the model. For deploying, we need to create a scoring script and also an environment which contains all the necessary libraries required to run. Luckily, Azure AutoML provides us with both files in the outputs folder. With both the scoring script and the environment, we can create a InferenceConfig file, and deploy the model along with using a deployment config file. I turn on Key based Authentication, and I also enable App Insights while deploying. 

![Deploy1](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/automl-deploy.png)

![Deploy2](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/deployed-model-active.png)
  
As we can see, the endpoint is deployed correctly and is active. The scoring file also contains an example of what the input data should look like. I try to create an example and query the endpoint. 

![Query1](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/endpoint-query-data.png)
  
![Query2](https://github.com/Neo96Mav/Azure-ML-Capstone/blob/main/Screenshots/endpoint-query-result.png)
  
The queried endpoint returns an output which means our code is working correctly. It can also be checked in the logs, which I also print. To finish the project, I delete the service. 
 
## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
