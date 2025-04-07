import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_entropy(y, sample_count=None):
    
    if sample_count is not None:
        # sklearn-ийн node_value-тай ажиллах үед
        total = sample_count.sum()
        if total == 0:
            return 0
        probabilities = sample_count.flatten() / total
        probabilities = probabilities[probabilities > 0]
    else:
        # Шууд ангилал утгатай ажиллах үед
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

def find_best_split(X, y, feature_index):
   
    feature_values = X.iloc[:, feature_index]
    unique_values = sorted(feature_values.unique())
    
    if len(unique_values) <= 1:
        return None, 0
    
    # Босго утгууд (хоёр утгын дундаж)
    thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]
    
    best_gain = -1
    best_threshold = None
    
    for threshold in thresholds:
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        
        # Хэт жижиг хуваалтуудыг алгасах
        if sum(left_mask) < 5 or sum(right_mask) < 5:
            continue
        
        gain = calculate_information_gain(
            y.values,
            y[left_mask].values,
            y[right_mask].values
        )
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    
    return best_threshold, best_gain

def visualize_entropy(proportions):
    
    # Энтропийг тооцоолох
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in proportions)
    
    # Дүрслэл
    plt.figure(figsize=(10, 6))
    labels = [f'Class {i+1}' for i in range(len(proportions))]
    colors = plt.cm.viridis(np.linspace(0, 1, len(proportions)))
    
    plt.bar(labels, proportions, color=colors)
    plt.axhline(y=1/len(proportions), color='r', linestyle='--', label='Тэнцүү магадлал')
    
    plt.title(f'Ангиллуудын магадлал, Энтропи = {entropy:.4f}')
    plt.xlabel('Ангилал')
    plt.ylabel('Магадлал')
    plt.legend()
    
    return entropy