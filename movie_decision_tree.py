import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math
import seaborn as sns

# Диаграммын загварыг тохируулах
plt.style.use('ggplot')
sns.set(style="whitegrid")

movies_df = pd.read_csv('movie_metadata.csv')

print(f"Өгөгдлийн сангийн хэмжээ: {movies_df.shape}")
print("\nЭхний 5 мөр:")
print(movies_df.head())

print("\nӨгөгдлийн сангийн бүтэц:")
print(movies_df.info())

print("\nТоон өгөгдлийн статистик:")
print(movies_df.describe())

print("\nХоосон утгуудын тоо:")
print(movies_df.isnull().sum())

selected_features = ['genres', 'budget', 'gross', 'imdb_score', 'title_year']
movies = movies_df[selected_features].copy()

print("Хоосон утгуудын тоо:")
print(movies.isnull().sum())

movies = movies.dropna()
print(f"\nХоосон утгуудыг устгасны дараах өгөгдлийн хэмжээ: {movies.shape}")

movies['success'] = (movies['imdb_score'] > 7).astype(int)
print("\nАмжилттай киноны тоо:", movies['success'].sum())
print("Амжилтгүй киноны тоо:", (movies['success'] == 0).sum())

# Жанрыг боловсруулах
# Жанрын жагсаалт үүсгэх
movies['genres'] = movies['genres'].str.split('|')

# Нэг өгөгдөл хэд хэдэн жанрт хамаарах тохиолдолд эхний жанрыг сонгох
movies['main_genre'] = movies['genres'].apply(lambda x: x[0])

# Жанрын тоо
print("\nЖанрын тоо:", movies['main_genre'].nunique())
print("Топ 10 жанр:")
print(movies['main_genre'].value_counts().head(10))

# Жанрыг кодлох (label encoding)
genre_encoder = LabelEncoder()
movies['genre_encoded'] = genre_encoder.fit_transform(movies['main_genre'])

# Шинж чанарууд ба зорилтот хувьсагчийг тодорхойлох
X = movies[['genre_encoded', 'budget', 'gross', 'title_year']]
y = movies['success']

# Өгөгдлийг сургалтын болон тестийн хэсэгт хуваах (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nСургалтын өгөгдлийн хэмжээ: {X_train.shape}")
print(f"Тестийн өгөгдлийн хэмжээ: {X_test.shape}")


# ## 4. Шийдвэрийн модны загвар сургах

dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

# Загварыг сургалтын өгөгдөл дээр тохируулах
dt_classifier.fit(X_train, y_train)

# Загварын үнэлгээ (сургалтын өгөгдөл дээр)
y_train_pred = dt_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Сургалтын өгөгдөл дээрх нарийвчлал: {train_accuracy:.4f}")

y_test_pred = dt_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Тестийн өгөгдөл дээрх нарийвчлал: {test_accuracy:.4f}")

print("\nАнгиллын тайлан:")
print(classification_report(y_test, y_test_pred))


# ## 5. Шийдвэрийн зангилаа тооцоолох

def calculate_entropy(y):
    
    if len(y) == 0:
        return 0
    
    # Ангилал тус бүрийн магадлалыг тооцоолох
    probabilities = np.bincount(y) / len(y)
    # 0 утгатай магадлалуудыг устгах (log2(0) тодорхойлогдоогүй)
    probabilities = probabilities[probabilities > 0]
    
    # Энтропийг тооцоолох
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(parent, left_child, right_child):
   
    parent_entropy = calculate_entropy(parent)
    
    # Хүүхэд зангилаануудын жин (өгөгдлийн тоогоор)
    weight_left = len(left_child) / len(parent)
    weight_right = len(right_child) / len(parent)
    
    # Жинлэсэн дундаж энтропи
    weighted_entropy = weight_left * calculate_entropy(left_child) + weight_right * calculate_entropy(right_child)
    
    # Мэдээллийн өсөлт
    information_gain = parent_entropy - weighted_entropy
    return information_gain

# Моделийн үндсэн шийдвэрийн зангилааг авах
tree_ = dt_classifier.tree_
feature = tree_.feature[0]  # Эхний зангилааны шинж чанарын индекс
threshold = tree_.threshold[0]  # Эхний зангилааны босго утга

# Шинж чанарын нэрийг авах
feature_name = X.columns[feature]
print(f"Эхний шийдвэрийн зангилаа: {feature_name} <= {threshold:.4f}")

# Эхний шийдвэрийн зангилааны мэдээллийн өсөлтийг тооцоолох
# Өгөгдлийг энэ зангилаагаар хуваах
mask = X_train.iloc[:, feature] <= threshold
left_indices = np.where(mask)[0]
right_indices = np.where(~mask)[0]

# Хуваалтын өмнөх ба дараах энтропи, мэдээллийн өсөлтийг тооцоолох
parent_entropy = calculate_entropy(y_train.values)
left_entropy = calculate_entropy(y_train.iloc[left_indices].values)
right_entropy = calculate_entropy(y_train.iloc[right_indices].values)

# Хуваалтын дараах жинлэсэн дундаж энтропи
weighted_entropy = (len(left_indices) / len(y_train)) * left_entropy + (len(right_indices) / len(y_train)) * right_entropy

# Мэдээллийн өсөлт
info_gain = parent_entropy - weighted_entropy

print(f"\nХуваалтын өмнөх энтропи: {parent_entropy:.4f}")
print(f"Зүүн хүүхэд зангилааны энтропи: {left_entropy:.4f} (n={len(left_indices)})")
print(f"Баруун хүүхэд зангилааны энтропи: {right_entropy:.4f} (n={len(right_indices)})")
print(f"Мэдээллийн өсөлт: {info_gain:.4f}")

# Бусад шинж чанаруудын мэдээллийн өсөлтийг тооцоолж хамгийн сайн шинж чанарыг батлах
print("\nБүх шинж чанаруудын мэдээллийн өсөлт:")

for i, feature_name in enumerate(X.columns):
    if i == feature:  # Аль хэдийн тооцоолсон шинж чанарыг алгасах
        print(f"{feature_name}: {info_gain:.4f} (загварын сонгосон шинж чанар)")
        continue
    
    # Боломжит босго утгуудыг авах (бодит өгөгдөлд хялбарчлахын тулд цөөн тооны утгыг авч үзэх)
    unique_values = np.percentile(X_train.iloc[:, i], [25, 50, 75])
    best_gain = 0
    best_threshold = None
    
    for threshold in unique_values:
        mask = X_train.iloc[:, i] <= threshold
        left_indices = np.where(mask)[0]
        right_indices = np.where(~mask)[0]
        
        # Хэт жижиг хуваалтуудыг алгасах
        if len(left_indices) < 5 or len(right_indices) < 5:
            continue
        
        left_labels = y_train.iloc[left_indices].values
        right_labels = y_train.iloc[right_indices].values
        
        gain = calculate_information_gain(y_train.values, left_labels, right_labels)
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    
    if best_threshold is not None:
        print(f"{feature_name}: {best_gain:.4f} (босго={best_threshold:.4f})")
    else:
        print(f"{feature_name}: Тохиромжтой босго утга олдсонгүй")


# Шийдвэрийн модыг зурах
plt.figure(figsize=(16, 10))
plot_tree(dt_classifier, 
          feature_names=X.columns, 
          class_names=['Амжилтгүй', 'Амжилттай'],
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title('Киноны амжилтыг урьдчилан таамаглах шийдвэрийн мод', fontsize=15)
plt.tight_layout()
plt.show()

# Шинж чанаруудын чухлыг харуулах
importance = dt_classifier.feature_importances_
indices = np.argsort(importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Чухлын зэрэг')
plt.ylabel('Шинж чанар')
plt.title('Шинж чанаруудын чухлын зэрэг')
plt.tight_layout()
plt.show()

# Киноны жанр болон амжилтын хоорондын хамаарал
plt.figure(figsize=(12, 6))
genre_success = movies.groupby('main_genre')['success'].mean().sort_values(ascending=False)
genre_counts = movies['main_genre'].value_counts()
top_genres = genre_counts[genre_counts > 50].index  # 50-с олон тохиолдолтой жанрууд

filtered_genre_success = genre_success[genre_success.index.isin(top_genres)]
sns.barplot(x=filtered_genre_success.index, y=filtered_genre_success.values)
plt.title('Жанр тус бүрийн амжилтын хувь (IMDB > 7)')
plt.xlabel('Жанр')
plt.ylabel('Амжилтын хувь')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Орлого, төсөв болон амжилтын хоорондын хамаарал
plt.figure(figsize=(12, 6))
plt.scatter(movies['budget'], movies['gross'], c=movies['success'], cmap='coolwarm', alpha=0.6)
plt.title('Төсөв, орлого болон амжилтын хоорондын хамаарал')
plt.xlabel('Төсөв')
plt.ylabel('Орлого')
plt.colorbar(label='Амжилт (1=Амжилттай, 0=Амжилтгүй)')
plt.tight_layout()
plt.show()

# IMDB үнэлгээний гистограмм
plt.figure(figsize=(10, 6))
sns.histplot(data=movies, x='imdb_score', hue='success', bins=20, kde=True)
plt.title('IMDB үнэлгээний тархалт')
plt.xlabel('IMDB үнэлгээ')
plt.ylabel('Киноны тоо')
plt.axvline(x=7, color='red', linestyle='--', label='Амжилтын босго (7)')
plt.legend()
plt.tight_layout()
plt.show()
