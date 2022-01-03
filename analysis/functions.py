import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import plotnine
from plotnine import *
#%matplotlib inline


def eastwest(df):
    map_dict = {1.0: "east",0.0:'west'}
    df['eastwest'] = df['eastwest'].map(map_dict)
    df['eastwest'] = df['eastwest'].fillna('unknown')
    return df


def check_na (df):
    check = df.isnull().sum()/len(df)*100
    return check


def remove_variables(df, drops):
    df = df.drop(drops, axis = 1)
    return df


def fillna(df):
    df.fillna(df.median(numeric_only=True, axis=0), axis=0, inplace=True)
    return df


def check_variance(df):
    var = df.var(numeric_only=True)
    return var


def distribution_plot(df):
    fig = plt.gcf()
    plt.rcParams["figure.figsize"] = (20,10)
    df.hist(bins=25, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
    plt.tight_layout(rect=(0, 0, 1.2, 1.2)) 
    return fig


def check_correlation(df):
    correlation = df.corr()
    return correlation


def plot_correlation_map(df):
    f, ax = plt.subplots(figsize=(30, 30))
    corr = df.corr()
    sns.set(font_scale=0.9)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    heatmap = sns.heatmap(round(corr,2), mask = mask, annot=True, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Correlation Heatmap', fontsize=14)
    return heatmap


def remove_corr_variables(df):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    # Find features with correlation greater than 0.7
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]
    # Drop features 
    df_1 = df.drop(to_drop, axis=1)
    return(df_1)


def feature_importance(df,dependant_variable):
    model = RandomForestRegressor(random_state=1, max_depth=100)
    model.fit(df,dependant_variable.values.ravel())
    fig = plt.figure(figsize=(10, 6))
    features = df.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[-9:]  # top 10 features
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    
def linear_model(df,dependant_variable):
    models = {}
    scores = {}
    for col in df.drop('eastwest', axis =1).columns:
        model_lr = LinearRegression()
        train = df[col].to_numpy()
        train = train.reshape(-1, 1)
        ln = model_lr.fit(train,dependant_variable) 
        scores[col] = model_lr.score(train,dependant_variable)
        models[col] = ln
    df_scores = pd.DataFrame.from_dict(scores, orient='index')
    df_scores = df_scores.rename(columns = {0:'R2'})
    df_scores = df_scores.sort_values(by = 'R2', ascending = False).head(10)
    return (df_scores, models)
 

def two_D_plot(df, x_value, y_value):
    fig = plt.figure(figsize=(15, 10))
    sns.regplot(y=x_value, x=y_value, data=df);
    
    
def multidimensional_plot(df):
    return ggplot(df, aes(y='eu_position', x='eu_budgets',color = 'galtan', size = 'multicult_dissent'))\
    + geom_point()\
    + facet_grid('~eastwest')\
    + labs(y='overall orientation of the party leadership towards European integration in 2019',\
           x='position of the party leadership in 2019 on EU authority over economic and budget policy.')\
    + theme(figure_size=(20, 8))\
    + geom_smooth(method='lm')


def fit(a,b,x, model_lr,df):
    a = float(np.asarray(model_lr.coef_))
    b = float(np.asarray(model_lr.intercept_))
    x = df['eu_budgets']
    y = a*x+b
    return (y)


def new_parties(y,b,a):
    new_parties_y = y.sample(10) ### y coordinate (eu_position values)
    new_parties_x = (new_parties_y-b)/a  ### calculate x coordinate for new parties (eu budget)
    new_parties_df = pd.DataFrame(list(zip(new_parties_x, new_parties_y))).rename(columns={0:'eu_budgets',1:'eu_position'})
    return new_parties_df


def new_parties_plot(df1, df2):
    g = (ggplot(df1) +
    geom_point(mapping=aes(x='eu_budgets', y='eu_position'),color = 'blue')+
    geom_point(df2,aes(x='eu_budgets', y='eu_position'), color = 'red', size = 4)
    )
    return g


def new_parties_features(feature, models):
    budgets_sample = np.array(feature).reshape(-1,1)
    predictions = {'eu_budgets':budgets_sample.reshape(-1)}
    for feature, model in models.items():
        predictions[feature] = model.predict(budgets_sample).reshape(-1)
    return pd.DataFrame(predictions)
