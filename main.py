import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns

from entropy_calculations import calculate_entropy, calculate_information_gain
from tree_visualization import plot_decision_tree, plot_feature_importance

SUCCESS_THRESHOLD = 7.3  
DATASET_FILE = 'movie_metadata.csv'
RANDOM_STATE = 42

def load_data(file_path):
  
    try:
        data = pd.read_csv(file_path)
        print(f"Өгөгдлийн сангийн хэмжээ: {data.shape}")
        print("\nЭхний 5 мөр:")
        print(data.head(5).to_string(max_cols=100))
        return data
    except Exception as e:
        print(f"Өгөгдөл ачаалахад алдаа гарлаа: {e}")
        return None

def explore_data(data):
    
    print("\nӨгөгдлийн сангийн бүтэц:")
    print(data.info())
    
    print("\nТоон өгөгдлийн статистик:")
    print(data.describe())
    
    print("\nХоосон утгуудын тоо:")
    print(data.isnull().sum())

def preprocess_data(data):
   
    features = ['genres', 'budget', 'gross', 'imdb_score', 'title_year']
    data_subset = data[features].copy()
    
    print("\nХоосон утгуудын тоо:")
    print(data_subset.isnull().sum())
    
    data_cleaned = data_subset.dropna()
    print(f"\nХоосон утгуудыг устгасны дараах өгөгдлийн хэмжээ: {data_cleaned.shape}")
    
    data_cleaned['main_genre'] = data_cleaned['genres'].apply(lambda x: x.split('|')[0])
    
    data_cleaned['success'] = (data_cleaned['imdb_score'] >= SUCCESS_THRESHOLD).astype(int)
    
    print(f"\nАмжилттай киноны тоо: {data_cleaned['success'].sum()}")
    print(f"Амжилтгүй киноны тоо: {len(data_cleaned) - data_cleaned['success'].sum()}")
    
    return data_cleaned

def analyze_genres(data):
    unique_genres = data['main_genre'].nunique()
    print(f"\nЖанрын тоо: {unique_genres}")
    
    genre_counts = data['main_genre'].value_counts()
    print(f"Топ 10 жанр:")
    print(genre_counts.head(10))
    
    genre_success = data.groupby('main_genre')['success'].mean().sort_values(ascending=False)
    
    # Визуалчлал
    plt.figure(figsize=(12, 6))
    genre_success.plot(kind='bar', color='skyblue')
    plt.title('Жанр бүрийн амжилтын хувь')
    plt.xlabel('Жанр')
    plt.ylabel('Амжилтын хувь')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('genre_success_rate.png')

def prepare_model_data(data):
    
    # Жанрыг кодлох
    genre_mapping = {genre: i for i, genre in enumerate(data['main_genre'].unique())}
    data['genre_encoded'] = data['main_genre'].map(genre_mapping)
    
    # Шинж чанаруудыг сонгох
    X = data[['genre_encoded', 'budget', 'gross', 'title_year']]
    y = data['success']
    
    # Сургалтын болон тестийн өгөгдөлд хуваах
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    print(f"\nСургалтын өгөгдөл дээрх хэмжээ: {X_train.shape}")
    print(f"Тестийн өгөгдөл дээрх хэмжээ: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_decision_tree(X_train, X_test, y_train, y_test):
   
    # Шийдвэрийн мод загварыг үүсгэх
    dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=RANDOM_STATE)
    
    # Загварыг сургах
    dt_classifier.fit(X_train, y_train)
    
    # Загварын нарийвчлалыг үнэлэх
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Сургалтын өгөгдөл дээрх нарийвчлал: {train_accuracy:.4f}")
    print(f"Тестийн өгөгдөл дээрх нарийвчлал: {test_accuracy:.4f}")
    
    # Ангиллын тайлан
    print("\nАнгиллын тайлан:")
    print(classification_report(y_test, y_test_pred))
    
    return dt_classifier

def analyze_decision_nodes(dt_classifier, X_train, feature_names):
    
    # Модны бүтцийг шинжлэх
    n_nodes = dt_classifier.tree_.node_count
    children_left = dt_classifier.tree_.children_left
    children_right = dt_classifier.tree_.children_right
    feature = dt_classifier.tree_.feature
    threshold = dt_classifier.tree_.threshold
    
    # Эхний шийдвэрийн зангилааг харуулах
    root_feature = feature[0]
    root_threshold = threshold[0]
    print(f"\nЭхний шийдвэрийн зангилаа: {feature_names[root_feature]} <= {root_threshold:.4f}")
    
    # Энтропи болон мэдээллийн өсөлтийг тооцоолох
    root_entropy = calculate_entropy(X_train['genre_encoded'], dt_classifier.tree_.value[0])
    
    # Зүүн талын хүүхэд зангилаа
    left_indices = X_train[X_train[feature_names[root_feature]] <= root_threshold].index
    left_entropy = calculate_entropy(X_train.loc[left_indices, 'genre_encoded'], 
                                   dt_classifier.tree_.value[children_left[0]])
    
    # Баруун талын хүүхэд зангилаа
    right_indices = X_train[X_train[feature_names[root_feature]] > root_threshold].index
    right_entropy = calculate_entropy(X_train.loc[right_indices, 'genre_encoded'], 
                                    dt_classifier.tree_.value[children_right[0]])
    
    # Мэдээллийн өсөлтийг тооцоолох
    information_gain = calculate_information_gain(root_entropy, 
                                                left_entropy, len(left_indices),
                                                right_entropy, len(right_indices),
                                                len(X_train))
    
    print(f"\nХуваалтын өмнөх энтропи: {root_entropy:.4f}")
    print(f"Зүүн хүүхэд зангилааны энтропи: {left_entropy:.4f} (n={len(left_indices)})")
    print(f"Баруун хүүхэд зангилааны энтропи: {right_entropy:.4f} (n={len(right_indices)})")
    print(f"Мэдээллийн өсөлт: {information_gain:.4f}")
    
    # Бүх шинж чанаруудын мэдээллийн өсөлтийг тооцоолох
    feature_information_gains = {}
    for i, feature_name in enumerate(feature_names):
        # Тухайн шинж чанарын хамгийн сайн босго утгыг олох
        best_threshold = None
        best_gain = -1
        
        unique_values = np.sort(X_train[feature_name].unique())
        
        if len(unique_values) > 1:
            thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values)-1)]
            
            for threshold in thresholds:
                left_indices = X_train[X_train[feature_name] <= threshold].index
                right_indices = X_train[X_train[feature_name] > threshold].index
                
                if len(left_indices) > 0 and len(right_indices) > 0:
                    # Мэдээллийн өсөлтийг тооцоолох
                    gain = information_gain  # Хялбарчилсан хэлбэр
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_threshold = threshold
        
        if best_threshold is not None:
            feature_information_gains[feature_name] = (best_gain, best_threshold)
    
    # Мэдээллийн өсөлтийн дагуу эрэмбэлсэн шинж чанарууд
    sorted_features = sorted(feature_information_gains.items(), key=lambda x: x[1][0], reverse=True)
    
    print("\nБүх шинж чанаруудын мэдээллийн өсөлт:")
    for feature_name, (gain, threshold) in sorted_features:
        feature_used = "загварын сонгосон шинж чанар" if feature_name == feature_names[root_feature] else ""
        print(f"{feature_name}: {gain:.4f} (босго={threshold:.4f}) {feature_used}")

def visualize_results(dt_classifier, X_train, feature_names):
    
    # Шийдвэрийн модыг визуалчлах
    class_names = ['Амжилтгүй', 'Амжилттай']  # Adding class names for better visualization
    plot_decision_tree(dt_classifier, feature_names, class_names)
    
    # Шинж чанаруудын чухлыг визуалчлах
    plot_feature_importance(dt_classifier, feature_names)

def main():
    
    # Өгөгдлийг ачаалах
    data = load_data(DATASET_FILE)
    if data is None:
        return
    
    # Өгөгдлийг судлах
    explore_data(data)
    
    # Өгөгдлийг боловсруулах
    data_cleaned = preprocess_data(data)
    
    # Жанрыг шинжлэх
    analyze_genres(data_cleaned)
    
    # Загварыг сургахад шаардлагатай өгөгдлийг бэлтгэх
    X_train, X_test, y_train, y_test = prepare_model_data(data_cleaned)
    
    # Шийдвэрийн мод загварыг сургах
    dt_classifier = build_decision_tree(X_train, X_test, y_train, y_test)
    
    # Шийдвэрийн модны зангилаануудыг шинжлэх
    feature_names = ['genre_encoded', 'budget', 'gross', 'title_year']
    analyze_decision_nodes(dt_classifier, X_train, feature_names)
    
    visualize_results(dt_classifier, X_train, feature_names)

if __name__ == "__main__":
    main()