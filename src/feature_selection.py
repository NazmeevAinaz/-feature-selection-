import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings

warnings.filterwarnings('ignore')

class StrokeFeatureSelection:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self):
        """Предобработка данных"""
        # Заполнение пропущенных значений
        self.df['bmi'] = self.df['bmi'].fillna(self.df['bmi'].median())
        
        # Удаление ненужных столбцов
        self.df = self.df.drop(['id'], axis=1)
        
        # Кодирование категориальных переменных
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            
        return self.df
    
    def prepare_features(self):
        """Подготовка признаков и целевой переменной"""
        X = self.df.drop('stroke', axis=1)
        y = self.df['stroke']
        
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_train_sel, X_test_sel, y_train, y_test):
        """Оценка качества модели"""
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        return accuracy_score(y_test, y_pred)
    
    def select_features_method(self, method_name, X_train, X_test, y_train, k):
        """Применяет различные методы отбора признаков"""
        if method_name == 'SelectKBest':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()]
            
        elif method_name == 'RFE':
            estimator = LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                C=0.1
            )
            selector = RFE(estimator, n_features_to_select=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()]
            
        elif method_name == 'RandomForest':
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            estimator.fit(X_train, y_train)
            importances = estimator.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            selected_features = X_train.columns[indices]
            X_train_sel = X_train.iloc[:, indices]
            X_test_sel = X_test.iloc[:, indices]
            
        elif method_name == 'Correlation':
            correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
            selected_features = correlations.head(k).index
            X_train_sel = X_train[selected_features]
            X_test_sel = X_test[selected_features]
            
        return X_train_sel, X_test_sel, selected_features
    
    def run_experiment(self):
        """Запуск полного эксперимента"""
        # Предобработка данных
        self.preprocess_data()
        
        # Подготовка признаков
        X_train, X_test, y_train, y_test = self.prepare_features()
        
        # Методы отбора признаков
        methods = ['SelectKBest', 'RFE', 'RandomForest', 'Correlation']
        max_features = min(8, X_train.shape[1])
        feature_range = range(1, max_features + 1)
        
        results = {method: [] for method in methods}
        selected_features_info = {method: [] for method in methods}
        
        print("Запуск экспериментов по отбору признаков...")
        
        for method in methods:
            print(f"Метод: {method}")
            for k in feature_range:
                try:
                    X_train_sel, X_test_sel, selected_features = self.select_features_method(
                        method, X_train, X_test, y_train, k
                    )
                    accuracy = self.evaluate_model(X_train_sel, X_test_sel, y_train, y_test)
                    results[method].append(accuracy)
                    selected_features_info[method].append(selected_features)
                    print(f"  Признаков: {k}, Accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"  Ошибка для k={k}: {e}")
                    results[method].append(np.nan)
                    selected_features_info[method].append([])
        
        # Базовый уровень (все признаки)
        baseline_accuracy = self.evaluate_model(X_train, X_test, y_train, y_test)
        
        return results, selected_features_info, baseline_accuracy, feature_range, methods
    
    def plot_results(self, results, baseline_accuracy, feature_range, methods):
        """Визуализация результатов"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D']
        
        for i, method in enumerate(methods):
            if len(results[method]) > 0 and not all(np.isnan(results[method])):
                plt.plot(feature_range, results[method], marker=markers[i], linewidth=2,
                         label=method, color=colors[i], markersize=8)
        
        plt.xlabel('Количество отобранных признаков', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Зависимость точности модели от количества отобранных признаков', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(feature_range)
        
        # Базовая линия (все признаки)
        plt.axhline(y=baseline_accuracy, color='red', linestyle='--', alpha=0.7,
                    label=f'Все признаки (Accuracy={baseline_accuracy:.4f})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/feature_selection_results.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    analyzer = StrokeFeatureSelection('data/healthcare-dataset-stroke-data.csv')
    results, selected_features_info, baseline_accuracy, feature_range, methods = analyzer.run_experiment()
    analyzer.plot_results(results, baseline_accuracy, feature_range, methods)
