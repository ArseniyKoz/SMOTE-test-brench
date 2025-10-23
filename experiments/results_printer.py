from typing import Dict, Any, List


class ResultsPrinter:

    def __init__(self):

        self.metric_names = {
            'accuracy': 'Accuracy',
            'balanced_accuracy': 'Balanced Accuracy',
            'f1_weighted': 'F1-weighted',
            'f1_macro': 'F1-macro',
            'g_mean': 'G-mean',
            'roc_auc': 'ROC AUC',
            'roc_auc_weighted': 'ROC AUC-weighted',
            'precision': 'Precision',
            'precision_weighted': 'Precision-weighted',
            'recall_weighted': 'Recall-weighted',
            'precision_macro': 'Precision-macro',
            'recall_macro': 'Recall-macro',
            'recall': 'Recall'
        }

        self.classifier_names = {
            'RandomForest': 'Random Forest',
            'SVM': 'SVM',
            'kNN': 'k-NN',
            'LogisticRegression': 'Logistic Regression',
            'DecisionTree': 'Decision Tree',
            'NaiveBayes': 'Naive Bayes'
        }

    def print_results(self, results: Dict[str, Any], config: Any = None) -> None:
        print("=" * 80)
        print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
        print("=" * 80)

        priority_metrics = self._get_priority_metrics(results, config)
        selected_classifiers = self._get_selected_classifiers(results, config)

        self._print_basic_info(results)
        self._print_config_info(config, priority_metrics, selected_classifiers)
        self._print_cv_results(results, priority_metrics, selected_classifiers)
        self._print_test_results(results, priority_metrics, selected_classifiers)
        print("=" * 80)

    def _get_priority_metrics(self, results: Dict[str, Any], config: Any) -> List[str]:

        if config and hasattr(config, 'priority_metrics'):
            return config.priority_metrics

        cv_results = results.get('cross_validation_results', {})
        if cv_results:
            first_classifier = list(cv_results.values())[0]
            available_metrics = []
            for key in first_classifier.keys():
                if key.endswith('_mean'):
                    metric_name = key.replace('_mean', '')
                    available_metrics.append(metric_name)
            return available_metrics

        return ['balanced_accuracy', 'f1_weighted', 'g_mean']

    def _get_selected_classifiers(self, results: Dict[str, Any], config: Any) -> List[str]:

        if config and hasattr(config, 'selected_classifiers'):
            return config.selected_classifiers

        cv_results = results.get('cross_validation_results', {})
        if cv_results:
            return list(cv_results.keys())

        final_results = results.get('final_test_results', {})
        if final_results:
            return list(final_results.keys())

        return []

    def _print_basic_info(self, results: Dict[str, Any]) -> None:
        metadata = results.get('metadata', {})
        dataset_info = results.get('dataset_info', {})

        print(f"Датасет: {metadata.get('dataset_name', 'N/A')}")
        print(f"Алгоритм: {metadata.get('algorithm_name', 'N/A')}")
        print(f"Время выполнения: {metadata.get('experiment_time', 0):.2f} сек")
        print(f"Дата: {metadata.get('timestamp', 'N/A')}")

        print(f"\nИнформация о данных:")
        print(f"  Всего образцов: {dataset_info.get('total_samples', 'N/A')}")
        print(f"  Признаков: {dataset_info.get('features', 'N/A')}")
        print(f"  Обучающих: {dataset_info.get('train_samples', 'N/A')}")
        print(f"  Тестовых: {dataset_info.get('test_samples', 'N/A')}")

        original_dist = dataset_info.get('original_class_distribution', [])
        if original_dist:
            print(f"  Распределение классов: {original_dist}")
            if len(original_dist) == 2:
                imbalance_ratio = max(original_dist) / min(original_dist) if min(original_dist) > 0 else float('inf')
                print(f"  Коэффициент дисбаланса: {imbalance_ratio:.2f}:1")

    def _print_config_info(self, config: Any, priority_metrics: List[str], selected_classifiers: List[str]) -> None:
        if config:
            print(f"\nКонфигурация эксперимента:")
            print(f"  Количество фолдов CV: {getattr(config, 'cv_folds', 'N/A')}")
            print(f"  Размер тестовой выборки: {getattr(config, 'test_size', 'N/A')}")
            print(f"  Random state: {getattr(config, 'random_state', 'N/A')}")

        print(f"\nОцениваемые метрики ({len(priority_metrics)}):")
        metric_display = [self.metric_names.get(m, m) for m in priority_metrics]
        print(f"  {', '.join(metric_display)}")

        print(f"\nИспользуемые классификаторы ({len(selected_classifiers)}):")
        classifier_display = [self.classifier_names.get(c, c) for c in selected_classifiers]
        print(f"  {', '.join(classifier_display)}")

    def _print_cv_results(self, results: Dict[str, Any], priority_metrics: List[str], selected_classifiers: List[str]) -> None:
        cv_results = results.get('cross_validation_results', {})

        print("\nРезультаты кросс-валидации:")
        print("-" * 100)

        header = f"{'Классификатор':<20}"
        for metric in priority_metrics:
            metric_display = self.metric_names.get(metric, metric)
            header += f" {metric_display:<18}"

        print(header)
        print("-" * 100)

        for clf_name in selected_classifiers:
            if clf_name not in cv_results:
                continue

            metrics = cv_results[clf_name]
            clf_display = self.classifier_names.get(clf_name, clf_name)

            row = f"{clf_display:<20}"

            for metric in priority_metrics:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'

                if mean_key in metrics and std_key in metrics:
                    mean_val = metrics[mean_key]
                    std_val = metrics[std_key]

                    if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                        metric_str = f"{mean_val:.3f}±{std_val:.3f}"
                    else:
                        metric_str = "N/A"
                else:
                    metric_str = "N/A"

                row += f" {metric_str:<18}"

            print(row)

        self._print_best_cv_results(cv_results, priority_metrics, selected_classifiers)

    def _print_best_cv_results(self, cv_results: Dict[str, Any], priority_metrics: List[str], selected_classifiers: List[str]) -> None:
        print("\nЛучшие результаты CV:")

        for metric in priority_metrics:
            metric_display = self.metric_names.get(metric, metric)
            mean_key = f'{metric}_mean'

            best_score = -1
            best_classifier = ""

            for clf_name in selected_classifiers:
                if clf_name in cv_results and mean_key in cv_results[clf_name]:
                    score = cv_results[clf_name][mean_key]
                    if isinstance(score, (int, float)) and score > best_score:
                        best_score = score
                        best_classifier = clf_name

            if best_classifier:
                clf_display = self.classifier_names.get(best_classifier, best_classifier)
                print(f"  {metric_display}: {clf_display} ({best_score:.3f})")

    def _print_test_results(self, results: Dict[str, Any], priority_metrics: List[str], selected_classifiers: List[str]) -> None:
        final_results = results.get('final_test_results', {})

        print("\nРезультаты на тестовой выборке:")
        print("-" * 80)

        for clf_name in selected_classifiers:
            if clf_name not in final_results:
                continue

            clf_data = final_results[clf_name]
            clf_display = self.classifier_names.get(clf_name, clf_name)

            print(f"\n{clf_display}:")

            original_data = clf_data.get('original_data', {})
            smote_data = clf_data.get('smote_data', {})
            improvements = clf_data.get('improvement', {})

            print(f"  {'Метрика':<20} {'Original':<12} {'SMOTE':<12} {'Изменение':<12}")
            print("  " + "-" * 56)

            positive_improvements = 0
            total_metrics = 0

            for metric in priority_metrics:
                if metric in original_data and metric in smote_data:
                    orig_val = original_data[metric]
                    smote_val = smote_data[metric]
                    improvement = improvements.get(metric, 0)

                    orig_str = f"{orig_val:.3f}" if isinstance(orig_val, (int, float)) else str(orig_val)
                    smote_str = f"{smote_val:.3f}" if isinstance(smote_val, (int, float)) else str(smote_val)

                    if isinstance(improvement, (int, float)):
                        if improvement > 0:
                            imp_str = f"+{improvement:.3f}"
                            positive_improvements += 1
                        else:
                            imp_str = f"{improvement:.3f}"
                        total_metrics += 1
                    else:
                        imp_str = "N/A"

                    metric_display = self.metric_names.get(metric, metric)
                    print(f"  {metric_display:<20} {orig_str:<12} {smote_str:<12} {imp_str:<12}")

            if total_metrics > 0:
                success_rate = positive_improvements / total_metrics * 100
                print(f"  Улучшено: {positive_improvements}/{total_metrics} ({success_rate:.1f}%)")

        self._print_overall_summary(final_results, priority_metrics, selected_classifiers)

    def _print_overall_summary(self, final_results: Dict[str, Any], priority_metrics: List[str], selected_classifiers: List[str]) -> None:
        total_improvements = 0
        total_comparisons = 0

        best_improvements = {}

        for clf_name in selected_classifiers:
            if clf_name not in final_results:
                continue

            improvements = final_results[clf_name].get('improvement', {})

            for metric in priority_metrics:
                if metric in improvements:
                    improvement = improvements[metric]
                    if isinstance(improvement, (int, float)):
                        total_comparisons += 1
                        if improvement > 0:
                            total_improvements += 1

                        if metric not in best_improvements or improvement > best_improvements[metric]['value']:
                            best_improvements[metric] = {
                                'value': improvement,
                                'classifier': clf_name
                            }

        if total_comparisons > 0:
            success_rate = total_improvements / total_comparisons * 100
            print(f"\nОбщая эффективность SMOTE:")
            print(f"  Улучшений: {total_improvements}/{total_comparisons} ({success_rate:.1f}%)")

            print(f"\nЛучшие улучшения:")
            for metric, data in best_improvements.items():
                if data['value'] > 0:
                    metric_display = self.metric_names.get(metric, metric)
                    clf_display = self.classifier_names.get(data['classifier'], data['classifier'])
                    print(f"  {metric_display}: +{data['value']:.3f} ({clf_display})")


def print_experiment_results(results: Dict[str, Any], config: Any = None) -> None:

    printer = ResultsPrinter()
    printer.print_results(results, config)

