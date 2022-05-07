# applied_ml_final_project

Jiayi Zhu

For the final project, I chose to do option #2 "Create a Publically Available Machine Learning Resource." 

I used the Maternity Health Risk Dataset (https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data) to train a model that can predict an individual's risk of health issues during pregnancy. Risk can either be low, mid, or high and takes into account factors such as age, blood sugar and pressure, body temperature and heartrate. The model that is used in my final app is a Random Forest Classsifier that achieved an average 85% accuracy when I ran k-fold cross validation on it. 

This repository is the same repository that is linked with Heroku for deployment of this machine learning resource. This repository also contains a Jupyter Notebook where I explored the dataset, prepared the dataset for training, tested several prediction algorithms, and determined the best one to use for the final app.

A few things I learned while doing this project include how to build a pipeline to prepare data for training and how to convert raw user input into data that my prediction model can understand and use. I also learned how to incorporate numerical input from an app UI into the app and how to deploy a streamlit app through Heroku CLI.

Link to the app: https://applied-ml-final-project.herokuapp.com/
