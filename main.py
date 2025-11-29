from src.feature_selection import StrokeFeatureSelection

def main():
    """Основная функция для запуска анализа"""
    print("Запуск анализа отбора признаков для предсказания инсульта...")
    
    # Инициализация анализатора
    analyzer = StrokeFeatureSelection('data/healthcare-dataset-stroke-data.csv')
    
    # Запуск эксперимента
    results, selected_features_info, baseline_accuracy, feature_range, methods = analyzer.run_experiment()
    
    # Вывод результатов
    print(f"\nБазовый accuracy (все признаки): {baseline_accuracy:.4f}")
    
    # Анализ лучших результатов
    for method in methods:
        if results[method]:
            best_acc = max([x for x in results[method] if not np.isnan(x)])
            best_k = results[method].index(best_acc) + 1
            print(f"{method}: Лучший accuracy {best_acc:.4f} при {best_k} признаках")

if __name__ == "__main__":
    main()
