# MMA Prediction Machine

This is an ongoing project used for predicting future UFC fights. 


## Data:

The project uses data from historical fights from the UFC organization, currently featuring data from 1990s to early 2024. Data was obtained from a Kaggle data set which was directly scraped from the UFC stats website. The dataset features usual features such as Age, Reach, Height etc. along with more detailed MMA specific features such as Strikes landed per minute, Takedown attempts per minute etc. Data from fights before the year 2003 has been culled, as the Unified Rules of MMA were implemented in that year. 

(Current, more refined datasets also exist, and this project is aiming to slowly pivot to those sets as well.)

Missing data for physical measurement data is imputed using the means of the respective feature and associated features. For example, missing reach values are imputed using the mean reach of other fighters in the same weight class within a 1 inch height range. 

### Data Representation


### Normalization



## Modelling:

The current main objective for this project is to find the most effective model for MMA fight predictions. The main models being tested are split between traditional models, such as Random Forest and XGBoost, deep learning models, such as RNN's and MLP's, and also a Transformer based model. 

### Pre-processing differences between Models


### Hyperparameter Tuning 


### Vector Embeddings

The Transformer model requires vector embeddings of the data, even though this task does not require NLP. Since the data for this project is in a tabular form, embedding and tokenization differentiates from commonly used techniques. There are many options to choose from, but this project will use TabTransformer with numerical data embeddings. 

With richer data sets, temporal structures can also be formed with respect to each individual fighter's historical data.