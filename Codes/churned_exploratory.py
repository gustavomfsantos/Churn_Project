# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:11:35 2024

@author: gusta
"""
#Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


from collections import Counter
import random
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)

def Load_Check_Data(data_path):
    
    # Import data
    df = pd.read_csv(data_path + "/Churn_Modelling.csv")
    
    # Display the first few rows of the dataset

    
    # Get a concise summary of the dataframe
    print(df.info())
    print('No Null Values')
    
    # Display summary statistics
    print(df.describe())
    
    #Check if CustomerID has repeated values
    duplicates = df['CustomerId'].duplicated().any()
    
    if duplicates:
        print("There are duplicated values in the ID column.")
    else:
        print("There are no duplicated values in the ID column.")
        
    #Drop Surname, and row Number, Cols with no value for analysis
    df.drop(['Surname', 'RowNumber'], inplace = True, axis =1)

    return df


save_param = True

def generate_plot(plt, save=save_param,  filename=None):
    """
    Generate a plot of x vs y and optionally save it to a file.

    Parameters:
        x (list or array-like): The x-values for the plot.
        y (list or array-like): The y-values for the plot.
        save (bool): Whether to save the plot or not. Default is False.
        filename (str): The filename to save the plot as. Required if save is True.
    """
    

    
    if save:
        if not filename:
            raise ValueError("Filename must be provided when save is True.")
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig(filename, dpi=100)
    
    #plt.show()


#Get eval from small models to have a Benchmark
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f'Accuracy: {acc:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
    return 

def Explore_Model_Data(df):
        
    # Count plot for churned clients
    sns.countplot(x='Exited', data=df)  
    generate_plot(plt, save=save_param, filename=plots_target + '/Target_Distribution.png')

    
    
    ###Should use any Balance Technique. Dataset is at 20%
    
    # List of numerical features
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    
    # Identify categorical features in your dataset
    categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    
    
    # Plotting distributions
    for feature in numerical_features:
        plt.clf()
        plt.figure(figsize=(10, 4))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        generate_plot(plt, save=save_param, filename=plots_features + f'/Distribution of {feature}')
        

    # Plot the distribution of each categorical feature
    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=feature, data=df)
        plt.title(f'Distribution of {feature}')
        generate_plot(plt, save=save_param, filename=plots_features + f'/{feature}_Distribution.png')

        
        
        
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(df[df['Exited'] == 0][feature], label='Not Churned', fill=True)
        sns.kdeplot(df[df['Exited'] == 1][feature], label='Churned', fill=True)
        plt.title(f'{feature} Distribution by Churn Status')
        plt.legend()
        generate_plot(plt, save=save_param, filename=plots_interaction + f'/{feature}_Distribution_by_Class.png')
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=feature, hue='Exited', data=df)
        plt.title(f'{feature} Distribution by Churn Status')
        generate_plot(plt, save=save_param, filename=plots_interaction + f'/{feature}_Distribution_by_Class.png')
    
    
    # Convert 'Geography' and 'Gender' into dummy variables
    # Set drop_first = True for regression models
    df_dummies = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False, dtype=int) 
    df_dummies.set_index('CustomerId', inplace = True)
   

        
    #Three Countries, Binary Columns for them. Binary Column for Gender
    # Calculate the correlation matrix for numerical features and the target variable 'Exited'
    corr_matrix = df_dummies.corr()
    
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    generate_plot(plt, save=save_param, filename=plots_interaction + f'/Correlation.png')
    #A pairplot can help visualize the relationships between pairs of variables in your dataset.
    # It's particularly useful for spotting correlations, patterns, and potential hypotheses about causal relationships.
    
    # Assuming 'Age', 'Tenure', 'Balance', 'NumOfProducts', and 'EstimatedSalary' are key numerical features
    features_to_plot = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Exited']
    sns.pairplot(df_dummies[features_to_plot], hue='Exited')
    generate_plot(plt, save=save_param, filename=plots_interaction + f'/PairPlot.png')
        
        
    #Understanding which features are most relevant to the target variable (in this case, 'Exited') can be critical. You can train a simple classifier like RandomForest and look at the feature importances it provides.
    
    
    
    # Preparing the data
    X = df_dummies.drop('Exited', axis=1)
    y = df_dummies['Exited']
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_proportions = y_train.value_counts(normalize=True)
    test_proportions = y_test.value_counts(normalize=True)
    
    # Create a DataFrame to make plotting easier
    distribution_df = pd.DataFrame({'Training Set': train_proportions, 'Testing Set': test_proportions})
    
    # Plotting
    distribution_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Distribution of Target Variable in Training and Testing Sets')
    plt.ylabel('Proportion')
    plt.xlabel('Class')
    plt.xticks(rotation=0)  # Keep the class labels horizontal
    generate_plot(plt, save=save_param, filename=plots_target + f'/Target_Distr_test_train.png')
    
    # Training the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Getting feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plotting feature importances
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.tight_layout()
    generate_plot(plt, save=save_param, filename=plots_features + f'/Features_Importance_RF.png')
    
    #Depending on your specific questions or hypotheses about the data, you might explore more advanced visualizations. For example, if you suspect that the relationship between age and churn varies by geography, you could visualize this with a facet grid.
    # Assuming 'Geography_France', 'Geography_Germany', and 'Geography_Spain' are the binary columns for geography
    for geography in ['Geography_France', 'Geography_Germany', 'Geography_Spain']:
        g = sns.FacetGrid(df_dummies[df_dummies[geography] == 1], hue="Exited", height=5)
        g.map(sns.scatterplot, "Age", "Balance", alpha=.7)
        g.add_legend()
        g.set_titles(geography)
        generate_plot(plt, save=save_param, filename=plots_interaction + f'/{geography}_Class_Distribution.png')
        
    #Identifying outliers can be crucial, especially in datasets where these outliers can represent anomalies or special cases that need further investigation.
    # Using boxplots to visualize potential outliers in the 'Age' feature
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Exited', y='Age', data=df)
    plt.title('Age Distribution by Churn Status')
    generate_plot(plt, save=save_param, filename=plots_features + f'/Age_Boxplot_By_Class.png')
    
    
    
    
    ###LOGIT
    # Standardize the features
    # Drop one class for each binary categorical transformed in order to avoid multicolinearity.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) #.drop(['Geography_Spain', 'Gender_Male'], axis = 1)
    X_test_scaled = scaler.transform(X_test) #.drop(['Geography_Spain', 'Gender_Male'], axis = 1)
    
    # Fit the logistic regression model
    logit_model = LogisticRegression(max_iter=1000)
    logit_model.fit(X_train_scaled, y_train)  #X_train_scaled
    
    # Get the coefficients
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': logit_model.coef_[0]})
    
    # Sort the coefficients for better visualization
    coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
    
    # Visualize the coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients)
    plt.title('Feature Coefficients from Logistic Regression')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    generate_plot(plt, save=save_param, filename=plots_features + f'/Features_Importance_Logit.png')
    
    
    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),     #X_train_scaled
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0) #X_train_scaled
    
    # Get model predictions as probabilities
    y_pred_probs_nn = model.predict(X_test_scaled) #X_test_scaled
    
    # Convert probabilities to binary class labels using a threshold of 0.5
    y_pred_nn = (y_pred_probs_nn > 0.5).astype(int).reshape(-1)


    
    
    # Evaluate the random forest model
    y_pred_rf = clf.predict(X_test)
    print("\nRandom Forest Model Evaluation:")
    cm_rf = evaluate_model(y_test, y_pred_rf)
    
    # Evaluate the logistic regression model
    y_pred_logit = logit_model.predict(X_test_scaled) # X_test_scaled
    print("Logistic Regression Model Evaluation:")
    cm_logit = evaluate_model(y_test, y_pred_logit)
    
    # Use the previously defined evaluation function to assess the neural network's performance
    print("Neural Network Model Evaluation:")
    cm_ann = evaluate_model(y_test, y_pred_nn)
    
    
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Random Forest')
    generate_plot(plt, save=save_param, filename=plots_results + f'/RF_Confusion_Matrix.png')
    
    
    
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_logit), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Logit')
    generate_plot(plt, save=save_param, filename=plots_results + f'/Logit_Confusion_Matrix.png')
    
    
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Neural Network')
    generate_plot(plt, save=save_param, filename=plots_results + f'/NN_Confusion_Matrix.png')
    

    return "Models With no Balancing Method"


##TRY SOME BALANCE TECHNIQUES! Recall too low
def Model_Balance_Classes(df):
    
    
    df_dummies = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=False, dtype=int) 
    df_dummies.set_index('CustomerId', inplace = True)
    
    
    # Preparing the data
    X = df_dummies.drop('Exited', axis=1)
    y = df_dummies['Exited']
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Dont change anything in test set
    
    train = pd.merge(X_train, y_train, on='CustomerId')
    
    train_churned = train[train['Exited'] == 1]
    train_not_churned = train[train['Exited'] == 0]
    
    print("Proportion in Training Set is", len(train_churned)/len(train_not_churned))
    print(pd.Series(train['Exited']).value_counts())
    size = len(train_not_churned)//3
    
    l = list(train_not_churned.index)
    
    def random_partition(seq, k):
        cnts = Counter(seq)

        while len(cnts) >= k:
            sample = random.sample(list(cnts), k)
            cnts -= Counter(sample)
            yield sample
    
    
        while cnts:
            sample = list(cnts)
            cnts -= Counter(sample)
            yield sample
        
    train_not_churned_lists = list(random_partition(l, size))
    
    train_1_not_churned = train_not_churned[train_not_churned.index.isin ( train_not_churned_lists[0])]
    train_2_not_churned = train_not_churned[train_not_churned.index.isin ( train_not_churned_lists[1])]
    train_3_not_churned = train_not_churned[train_not_churned.index.isin ( train_not_churned_lists[2])]
    

    
    # Find common indexes between df1 and df2
    common_indexes_12 = train_1_not_churned.index.intersection(train_2_not_churned.index)
    
    # Find common indexes between the result above and df3
    common_indexes_all = common_indexes_12.intersection(train_3_not_churned.index)
    
    # Check if there are any common indexes
    if not common_indexes_all.empty:
        print("There are common indexes:")
        print(common_indexes_all)
    else:
        print("There are no common indexes among the three dataframes.")
        
    #Merge with the churned set now     
    final_train1 = pd.concat([train_1_not_churned, train_churned], axis=0).sort_values('CustomerId')
    final_train2 = pd.concat([train_2_not_churned, train_churned], axis=0).sort_values('CustomerId')
    final_train3 = pd.concat([train_3_not_churned, train_churned], axis=0).sort_values('CustomerId')
    
    for df_aux in [final_train1, final_train2, final_train3]:
        churned = df_aux[df_aux['Exited'] == 1]
        not_churned = df_aux[df_aux['Exited'] == 0]
        
        print("Proportion in Training Set is", len(churned)/len(not_churned))
        print(pd.Series(df_aux['Exited']).value_counts())
    #Classes are better balanced
    
    X_train1 = final_train1.drop('Exited', axis=1)
    y_train1 = final_train1['Exited']
    
    X_train2 = final_train2.drop('Exited', axis=1)
    y_train2 = final_train2['Exited']
    
    X_train3 = final_train3.drop('Exited', axis=1)
    y_train3 = final_train3['Exited']
    
    #Scaler trained in full Train set and apllied to each train and the full Test Set
    
    scaler = StandardScaler()
    scaler.fit(X_train) #.drop(['Geography_Spain', 'Gender_Male'], axis = 1)
    X_test_scaled = scaler.transform(X_test) #.drop(['Geography_Spain', 'Gender_Male'], axis = 1)
        
    X_train1_transformed = scaler.transform(X_train1)
    X_train2_transformed = scaler.transform(X_train2)
    X_train3_transformed = scaler.transform(X_train3)
    
    
    logit_model1 = LogisticRegression(max_iter=1000).fit(X_train1_transformed, y_train1)
    logit_model2 = LogisticRegression(max_iter=1000).fit(X_train2_transformed, y_train2)
    logit_model3 = LogisticRegression(max_iter=1000).fit(X_train3_transformed, y_train3)

    rf_model1 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train1, y_train1)
    rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train2, y_train2)
    rf_model3 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train3, y_train3)
    
    nn_model1 = Sequential([
        Dense(128, activation='relu', input_shape=(X_train1_transformed.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model1.fit(X_train1_transformed, y_train1, epochs=100, batch_size=32, validation_split=0.2, verbose=0) 
        
    nn_model2 = Sequential([
        Dense(128, activation='relu', input_shape=(X_train2_transformed.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
               )
    nn_model2.fit(X_train2_transformed, y_train2, epochs=100, batch_size=32, validation_split=0.2, verbose=0) 
               
    nn_model3 = Sequential([
        Dense(128, activation='relu', input_shape=(X_train3_transformed.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
               )
    nn_model3.fit(X_train3_transformed, y_train3, epochs=100, batch_size=32, validation_split=0.2, verbose=0) 
               
               
               
    def ensemble_vote(models, X, neural_network = False):
    # Get predictions from all models
        if neural_network == True:
            y_pred_probs_nn1 = models[0].predict(X)
            y_pred_probs_nn2 = models[1].predict(X)
            y_pred_probs_nn3 = models[2].predict(X)
            # Convert probabilities to binary class labels
            y_pred_nn1 = (y_pred_probs_nn1 > 0.5).astype(int).reshape(-1) 
            y_pred_nn2 = (y_pred_probs_nn2 > 0.5).astype(int).reshape(-1) 
            y_pred_nn3 = (y_pred_probs_nn3 > 0.5).astype(int).reshape(-1) 
            
            predictions = np.array((y_pred_nn1, y_pred_nn2, y_pred_nn3))
            
        else:
            
            predictions = np.array([model.predict(X) for model in models])
            
           
            # Transpose predictions to get them in shape (samples, models)
        predictions = predictions.T
        
        # Use majority vote for final prediction
        # For each sample, count the occurrences of each class and use the most frequent one
        final_predictions = [np.argmax(np.bincount(sample_predictions)) for sample_predictions in predictions]
        
        return final_predictions

    # Assuming X_test is your test data
    final_logit = ensemble_vote([logit_model1, logit_model2, logit_model3], X_test)
    
    final_rm = ensemble_vote([rf_model1, rf_model2, rf_model3], X_test)
    
    final_nn = ensemble_vote([nn_model1, nn_model2, nn_model3], X_test, neural_network = True)
    
    
    
    print('Logit Ensemble')
    (evaluate_model(y_test, final_logit))
    
    print('RandomForest Ensemble')
    (evaluate_model(y_test, final_rm))
    
    print('Neural Network Ensemble')
    (evaluate_model(y_test, final_nn))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print('Doing Balacing using imblearn Package.')
    
    
    # # Initialize the RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)

    
    X_train_scaled = scaler.transform(X_train)
    # Fit and apply the transformation
    X_resampled, y_resampled = rus.fit_resample(X_train_scaled, y_train)
    
    # Check the new class distribution
    print(pd.Series(y_resampled).value_counts())
    
    
    
    
    # Initialize the Logistic Regression model
    logit_model_resampled = LogisticRegression(max_iter=1000)
    
    # Train the model
    logit_model_resampled.fit(X_resampled, y_resampled)
    
    # Predictions on the test set
    y_pred_logit = logit_model_resampled.predict(X_test_scaled)
    

    
    
    # Initialize the Random Forest model
    rf_model_resampled = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf_model_resampled.fit(X_resampled, y_resampled)
    
    # Predictions on the test set
    y_pred_rf = rf_model_resampled.predict(X_test_scaled)
    
    
    # Define the neural network model
    nn_model_resampled = Sequential([
        Dense(128, activation='relu', input_shape=(X_resampled.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    nn_model_resampled.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    nn_model_resampled.fit(X_resampled, y_resampled, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Predictions on the test set (probabilities)
    y_pred_probs_nn = nn_model_resampled.predict(X_test_scaled)
    
    # Convert probabilities to binary class labels
    y_pred_nn = (y_pred_probs_nn > 0.5).astype(int).reshape(-1)
    
    
    # Evaluation
    print("Logistic Regression (Under Sample) Evaluation:")
    (evaluate_model(y_test, y_pred_logit))

    
    # Evaluation
    print("Random Forest (Under Sampled) Evaluation:")
    (evaluate_model(y_test, y_pred_rf))

    
    
    # Evaluation
    print("Neural Network (Under Sample) Evaluation:")
    (evaluate_model(y_test, y_pred_nn))

    
    
    
    return "Models Balanced"



def save_console_output(filename):
    """
    Save the console output to a file.

    Parameters:
        filename (str): The name of the file to save the output to.
    """
    original_stdout = sys.stdout
    with open(filename, 'w') as file:
        sys.stdout = file
        # Your code here
        print("Hello, world!")  # Example output
    sys.stdout = original_stdout

# Example usage:
save_console_output("console_output.txt")

# This part checks if the script is the main program and runs the functions if it is
if __name__ == "__main__":
    
    warnings.filterwarnings('ignore')

    #Paths
    project_path = r"C:\Users\gusta\OneDrive\Área de Trabalho\Personal_Projects\Churn_Classic_Project"

    console_path = project_path + '/Console_Logs'
    data_path = project_path + "/Data"
    plots_path = project_path + "/Plots"
    
    plots_target = plots_path + "/Target"
    plots_features = plots_path + "/Features"
    plots_interaction = plots_path + "/Interaction"
    plots_results = plots_path + "/Results"
    
    save_param = True
    
    df = Load_Check_Data(data_path)
    
    Explore_Model_Data(df)
    
    Model_Balance_Classes(df)

    save_console_output(console_path + '/console_log')