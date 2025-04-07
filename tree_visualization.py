import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

def plot_decision_tree(model, feature_names, class_names, figsize=(16, 10)):
    
    plt.figure(figsize=figsize)
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=class_names,
              filled=True, 
              rounded=True, 
              fontsize=10)
    plt.title('Киноны амжилтыг урьдчилан таамаглах шийдвэрийн мод', fontsize=15)
    plt.tight_layout()
    
def plot_feature_importance(model, feature_names, figsize=(10, 6)):
    
    importance = model.feature_importances_
    indices = np.argsort(importance)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Чухлын зэрэг')
    plt.ylabel('Шинж чанар')
    plt.title('Шинж чанаруудын чухлын зэрэг')
    plt.tight_layout()
    
def plot_genre_success_rate(movies, min_count=50, figsize=(12, 6)):
  
    plt.figure(figsize=figsize)
    genre_success = movies.groupby('main_genre')['success'].mean().sort_values(ascending=False)
    genre_counts = movies['main_genre'].value_counts()
    top_genres = genre_counts[genre_counts > min_count].index
    
    filtered_genre_success = genre_success[genre_success.index.isin(top_genres)]
    sns.barplot(x=filtered_genre_success.index, y=filtered_genre_success.values)
    plt.title('Жанр тус бүрийн амжилтын хувь (IMDB > 7)')
    plt.xlabel('Жанр')
    plt.ylabel('Амжилтын хувь')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
def plot_budget_gross_success(movies, figsize=(12, 6)):
    
    plt.figure(figsize=figsize)
    plt.scatter(movies['budget'], movies['gross'], c=movies['success'], cmap='coolwarm', alpha=0.6)
    plt.title('Төсөв, орлого болон амжилтын хоорондын хамаарал')
    plt.xlabel('Төсөв')
    plt.ylabel('Орлого')
    plt.colorbar(label='Амжилт (1=Амжилттай, 0=Амжилтгүй)')
    plt.tight_layout()
    
def plot_imdb_score_distribution(movies, figsize=(10, 6)):
    
    plt.figure(figsize=figsize)
    sns.histplot(data=movies, x='imdb_score', hue='success', bins=20, kde=True)
    plt.title('IMDB үнэлгээний тархалт')
    plt.xlabel('IMDB үнэлгээ')
    plt.ylabel('Киноны тоо')
    plt.axvline(x=7, color='red', linestyle='--', label='Амжилтын босго (7)')
    plt.legend()
    plt.tight_layout() 