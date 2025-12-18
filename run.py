import data_parser
import graphing
import regression
import decision_tree
import neural_network
import metrics

import numpy as np

from pathlib import Path
import shutil

OUT_DIR = "./data"

dir_path = Path(OUT_DIR)

if dir_path.is_dir():
    try:
        shutil.rmtree(dir_path)
        print(f"[LOG]: Folder '{dir_path}' and all its contents have been deleted.")
    except OSError as e:
        print(f"Error: {e.strerror}")
else:
    print(f"[LOG]: Folder '{dir_path}' does not exist or is not a directory.")

REGRESSION_DATA_PATH = "./project_files/regression_data.csv"
CLASSIFICATION_DATA_PATH = "./project_files/classification_data.csv"

train_regression_X, train_regression_Y, test_regression_X, test_regression_Y = data_parser.parse_data(REGRESSION_DATA_PATH, data_fields=(0,), target_fields=(8,), string_fields=tuple())
train_biased_regression_X = data_parser.prepare_X(train_regression_X)
test_biased_regression_X = data_parser.prepare_X(test_regression_X)

train_classification_X, train_classification_Y, test_classification_X, test_classification_Y = data_parser.parse_data(CLASSIFICATION_DATA_PATH, data_fields=(0,), target_fields=(7,), string_fields=tuple())
train_biased_classification_X = data_parser.prepare_X(train_classification_X)
test_biased_classification_X = data_parser.prepare_X(test_classification_X)

def get_params(model, result=None):
    if result == None:
        result = {}
    
    REMOVE_FIELDS = ("X", "Y", "y", "_theta", "_w", "_alpha_vec_", "_X_train_scaled_", "loss_history_", "nll_history_", "X_mean_", "X_std_", "x_mean_", "x_std_", "feature_powers_", "phi_mean_", "phi_std_", "y_mean_", "y_std_", "_Z_train_scaled_", "_y_mean_rbf_", "_y_std_rbf_", "X_train_", "y_train_", "alphas_", "K_", "base_estimator", "_estimators", "_feature_indices", "_rng", "W", "b", "W_out", "b_out", "W_reg", "b_reg", "W_clf", "b_clf", "centers_", "gamma_", "convs", "pools", "flatten", "rnn", "_opt", "_reg", "_act", "_regularizer")
    params = model.__dict__.copy()

    for i in REMOVE_FIELDS:
        params.pop(i, None)

    for i in result:
        if i != "roc":
            params[i] = result[i]
    
    return params

plotter = graphing.Plotter()

if False: # Plots the graphs... important to turn on for final
    plotter.plot_regression(train_regression_X, train_regression_Y)
    plotter.plot_classification(train_classification_X, train_classification_Y)

correlation = metrics.FeatureCorrelation(train_regression_X, train_regression_Y)
print(f"[LOG]: {correlation.result()}")
correlation = metrics.FeatureCorrelation(train_classification_X, train_classification_Y)
print(f"[LOG]: {correlation.result()}")

regression_metric = metrics.RegressionEvaluation(test_regression_X, test_regression_Y, None)
classification_metric = metrics.ClassificationEvaluation(test_classification_X, test_classification_Y, test_classification_Y)

# 1. REGRESSION

# Linear Regression
model = regression.LinearRegression(train_biased_regression_X, train_regression_Y)

methods = ("closed_form", "gd", "rbf")
optimizers = ("vanilla", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam")
modes = ("batch", "stochastic", "minibatch")
penalties = ("none", "ridge", "lasso", "elasticnet")

for method in methods:
    model.method = method

    if method == "gd":
        for optimizer in optimizers:
            for mode in modes:
                for penalty in penalties:
                    model.optimizer = optimizer
                    model.batch_mode = mode
                    model.penalty = penalty

                    print(f"[LOG]: Linear Regression - Gradient Descent - {optimizer.title()} - {mode.title()} - {penalty.title()}")
                    model.fit()
                    data_parser.save_run_artifact(OUT_DIR, model._theta, get_params(model), "theta", run_id_prefix=f"run_LinearRegression_{model.method}_{model.optimizer}_{model.batch_mode}_{model.penalty}_")
                    regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
                    data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_LinearRegression_{model.method}_{model.optimizer}_{model.batch_mode}_{model.penalty}_")
    elif method == "closed_form":
        for penalty in penalties:
            model.penalty = penalty

            print(f"[LOG]: Linear Regression - Closed Form - {penalty.title()}")
            model.fit()
            data_parser.save_run_artifact(OUT_DIR, model._theta, get_params(model), "theta", run_id_prefix=f"run_LinearRegression_{model.method}_{model.penalty}_")
            regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_LinearRegression_{model.method}_{model.penalty}_")
    else:
        print(f"[LOG]: Linear Regression - RBF")

        model.fit()
        data_parser.save_run_artifact(OUT_DIR, model._alpha_vec_, get_params(model), "alpha", run_id_prefix=f"run_LinearRegression_{model.method}_")
        regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
        data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_LinearRegression_{model.method}_")

# Logistic Regression
model = regression.LogisticRegression(train_biased_classification_X, train_classification_Y)

methods = ("newton", "gd", "rbf")
optimizers = ("vanilla", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam")
modes = ("batch", "stochastic", "minibatch")
penalties = ("none", "ridge", "lasso", "elasticnet")

for method in methods:
    model.method = method

    if method == "gd":
        for optimizer in optimizers:
            for mode in modes:
                for penalty in penalties:
                    model.optimizer = optimizer
                    model.batch_mode = mode
                    model.penalty = penalty

                    print(f"[LOG]: Logistic Regression - Gradient Descent - {optimizer.title()} - {mode.title()} - {penalty.title()}")
                    model.fit()
                    data_parser.save_run_artifact(OUT_DIR, model._w, get_params(model), "w", run_id_prefix=f"run_LogisticRegression_{model.method}_{model.optimizer}_{model.batch_mode}_{model.penalty}_")
                    classification_metric._classes = np.unique(train_classification_Y)
                    classification_metric._Y_prob = prob = model.predict_proba(test_biased_classification_X)
                    data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), "probabilities", run_id_prefix=f"run_LogisticRegression_{model.method}_{model.optimizer}_{model.batch_mode}_{model.penalty}_")
                    prediction = model.predict(test_biased_classification_X)
                    data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_LogisticRegression_{model.method}_{model.optimizer}_{model.batch_mode}_{model.penalty}_")
    elif method == "newton":
        for penalty in ("none", "ridge"):
            model.penalty = penalty

            print(f"[LOG]: Logistic Regression - Newton - {penalty.title()}")
            model.fit()
            data_parser.save_run_artifact(OUT_DIR, model._w, get_params(model), "w", run_id_prefix=f"run_LogisticRegression_{model.method}_{model.penalty}_")
            classification_metric._classes = np.unique(train_classification_Y)
            classification_metric._Y_prob = prob = model.predict_proba(test_biased_classification_X)
            data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), "probabilities", run_id_prefix=f"run_LogisticRegression_{model.method}_{model.penalty}_")
            prediction = model.predict(test_biased_classification_X)
            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_LogisticRegression_{model.method}_{model.penalty}_")
    else:
        model.alpha = 0.5

        print(f"[LOG]: Logistic Regression - RBF")
        model.fit()
        data_parser.save_run_artifact(OUT_DIR, model._alpha_vec_, get_params(model), "alpha", run_id_prefix=f"run_LogisticRegression_{model.method}_")
        classification_metric._classes = np.unique(train_classification_Y)
        classification_metric._Y_prob = prob = model.predict_proba(test_biased_classification_X)
        data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), "probabilities", run_id_prefix=f"run_LogisticRegression_{model.method}_")
        prediction = model.predict(test_biased_classification_X)
        data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_LogisticRegression_{model.method}_")

# Polynomial Regression
model = regression.PolynomialRegression(train_biased_regression_X, train_regression_Y)

degrees = (2, 3, 5, 10)
methods = ("closed_form", "cd", "gd", "rbf")
optimizers = ("vanilla", "momentum", "nesterov", "adagrad", "rmsprop", "adadelta", "adam")
modes = ("batch", "stochastic", "minibatch")

for degree in degrees:
    model.degree = degree

    for method in methods:
        model.method = method

        if method == "closed_form":
            for penalty in ("none", "ridge"):
                model.penalty = penalty

                print(f"[LOG]: Polynomial Regression - degree={degree} - Closed Form - {penalty.title()}")
                model.fit()
                data_parser.save_run_artifact(OUT_DIR, model._w, get_params(model), "w", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_{model.penalty}_")
                regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
                data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_{model.penalty}_")
        elif method == "cd":
            for penalty in ("lasso", "elasticnet"):
                model.penalty = penalty

                print(f"[LOG]: Polynomial Regression - degree={degree} - Coordinate Descent - {penalty.title()}")
                model.fit()
                data_parser.save_run_artifact(OUT_DIR, model._w, get_params(model), "w", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_{model.penalty}_")
                regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
                data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_{model.penalty}_")
        elif method == "gd":
            for optimizer in optimizers:
                for mode in modes:
                    for penalty in ("none", "ridge"):
                        model.optimizer = optimizer
                        model.batch_mode = mode
                        model.penalty = penalty

                        print(f"[LOG]: Polynomial Regression - degree={degree} - Gradient Descent - {optimizer.title()} - {mode.title()} - {penalty.title()}")
                        model.fit()
                        data_parser.save_run_artifact(OUT_DIR, model._w, get_params(model), "w", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_{model.optimizer}_{model.batch_mode}_{model.penalty}_")
                        regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
                        data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_{model.optimizer}_{model.batch_mode}_{model.penalty}_")
        else:
            print(f"[LOG]: Polynomial Regression - degree={degree} - RBF")
            model.fit()
            data_parser.save_run_artifact(OUT_DIR, model._alpha_vec_, get_params(model), "alpha", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_")
            regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_PolynomialRegression_deg{degree}_{model.method}_")

# 2. DECISION TREE

# Regression Decision Tree
model = decision_tree.RegressionDecisionTree()

split_types = ("id3", "c45", "cart")

for split_type in split_types:
    model.split_type = split_type

    if split_type == "id3":
        for criterion in ("entropy", "gini"):
            model.criterion = criterion

            print(f"[LOG]: Regression Decision Tree - {model.split_type.upper()} - {model.criterion.upper()}")
            model.fit(train_regression_X, train_regression_Y)
            data_parser.save_run_tree(OUT_DIR, model._root, get_params(model), artifact_name="tree", run_id_prefix=f"run_RegressionDecisionTree_{model.split_type.upper()}_{model.criterion.title()}_")
            regression_metric.Y_pred = prediction = model.predict(test_regression_X)
            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_RegressionDecisionTree_{model.split_type.upper()}_{model.criterion.title()}_")
    elif split_type == "c45":
        model.criterion = "gain_ratio"

        print(f"[LOG]: Regression Decision Tree - {model.split_type.upper()} - {model.criterion.upper()}")
        model.fit(train_regression_X, train_regression_Y)
        data_parser.save_run_tree(OUT_DIR, model._root, get_params(model), artifact_name="tree", run_id_prefix=f"run_RegressionDecisionTree_{model.split_type.upper()}_")
        regression_metric.Y_pred = prediction = model.predict(test_regression_X)
        data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_RegressionDecisionTree_{model.split_type.upper()}_")
    else:
        model.criterion = "mse"

        print(f"[LOG]: Regression Decision Tree - {model.split_type.upper()} - {model.criterion.upper()}")
        model.fit(train_regression_X, train_regression_Y)
        data_parser.save_run_tree(OUT_DIR, model._root, get_params(model), artifact_name="tree", run_id_prefix=f"run_RegressionDecisionTree_{model.split_type.upper()}_")
        regression_metric.Y_pred = prediction = model.predict(test_regression_X)
        data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), "prediction", run_id_prefix=f"run_RegressionDecisionTree_{model.split_type.upper()}_")

# Decision Tree Classifier
model = decision_tree.DecisionTreeClassifier()

split_types = ("id3", "c45", "cart")

for split_type in split_types:
    model.split_type = split_type

    if split_type == "id3":
        for criterion in ("entropy", "gini"):
            model.criterion = criterion

            print(f"[LOG]: Decision Tree Classifier - {model.split_type.upper()} - {model.criterion.upper()}")
            model.fit(train_classification_X, train_classification_Y)
            data_parser.save_run_tree(OUT_DIR, model._root, get_params(model), artifact_name="tree", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_{model.criterion.title()}_")
            classification_metric._classes = model._classes
            classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
            data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), "probabilities", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_{model.criterion.title()}_")
            prediction = model.predict(test_classification_X)
            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_{model.criterion.title()}_")
    elif split_type == "c45":
        model.criterion = "gain_ratio"

        print(f"[LOG]: Decision Tree Classifier - {model.split_type.upper()} - {model.criterion.upper()}")
        model.fit(train_classification_X, train_classification_Y)
        data_parser.save_run_tree(OUT_DIR, model._root, get_params(model), artifact_name="tree", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_")
        classification_metric._classes = model._classes
        classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
        data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), "probabilities", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_")
        prediction = model.predict(test_classification_X)
        data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_{model.criterion.title()}_")
    else:
        for criterion in ("gini", "entropy"):
            model.criterion = criterion

            print(f"[LOG]: Decision Tree Classifier - {model.split_type.upper()} - {model.criterion.upper()}")
            model.fit(train_classification_X, train_classification_Y)
            data_parser.save_run_tree(OUT_DIR, model._root, get_params(model), artifact_name="tree", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_{model.criterion.title()}_")
            classification_metric._classes = model._classes
            classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
            data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), "probabilities", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_{model.criterion.title()}_")
            prediction = model.predict(test_classification_X)
            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_DecisionTreeClassifier_{model.split_type.upper()}_{model.criterion.title()}_")

# SVM Classifier
model = decision_tree.SVMClassifier()

kernels = ("linear", "rbf")

for kernel in kernels:
    model.kernel = kernel

    print(f"[LOG]: SVM Classifier - {model.kernel.upper()}")
    model.fit(train_classification_X, train_classification_Y)

    if kernel == "linear":
        data_parser.save_run_artifact(OUT_DIR, model._w, get_params(model), artifact_name="weights", run_id_prefix=f"run_SVMClassifier_{model.kernel.title()}_")
        data_parser.save_run_artifact(OUT_DIR, model._b, get_params(model), artifact_name="bias", run_id_prefix=f"run_SVMClassifier_{model.kernel.title()}_")
    else:
        data_parser.save_run_artifact(OUT_DIR, model.alphas_, get_params(model), artifact_name="alphas", run_id_prefix=f"run_SVMClassifier_{model.kernel.title()}_")
        data_parser.save_run_artifact(OUT_DIR, model.b_, get_params(model), artifact_name="bias", run_id_prefix=f"run_SVMClassifier_{model.kernel.title()}_")
    
    classification_metric._classes = model._classes
    classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
    data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), "probabilities", run_id_prefix=f"run_SVMClassifier_{model.kernel.title()}_")
    prediction = model.predict(test_classification_X)
    data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_SVMClassifier_{model.kernel.title()}_")

# Ensemble Learning
methods = ("bagging", "adaboost")
feature_selections = ("none", "subspace", "mask")
n_values = (2, 3, 5, 10)
base_models = ("DecisionTreeClassifier", "SVMClassifier")

for base_model_name in base_models:
    for method in methods:
        for feature_selection in feature_selections:
            for n in n_values:
                if base_model_name == "DecisionTreeClassifier":
                    split_types = ("id3", "c45", "cart")

                    for split_type in split_types:
                        if split_type == "id3":
                            for criterion in ("entropy", "gini"):
                                base_model = decision_tree.DecisionTreeClassifier(split_type=split_type, criterion=criterion)
                                model = decision_tree.EnsembleLearning(base_estimator=base_model, method=method, n_estimators=n, max_samples=1.0, max_features=1.0, feature_selection=feature_selection, mask_keep_prob=0.7, oob_score=(method == "bagging"), random_state=None)

                                print(f"[LOG]: Ensemble Learning - n={n} - DecisionTreeClassifier - {split_type.upper()} - {base_model.criterion.upper()} - {method.upper()} - {feature_selection.upper()}")
                                model.fit(train_classification_X, train_classification_Y)
                                data_parser.save_run_artifact(OUT_DIR, np.array(model._feature_indices, dtype=object), get_params(model), artifact_name="feature_indices", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                                if method == "adaboost":
                                    data_parser.save_run_artifact(OUT_DIR, np.array(model._estimator_weights, dtype=float), get_params(model), artifact_name="estimator_weights", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                                data_parser.save_run_tree(OUT_DIR, [est._root for est in model._estimators], get_params(model), artifact_name="trees", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")

                                classification_metric._classes = np.unique(train_classification_Y)
                                classification_metric._Y_prob = prob =  model.predict_proba(test_classification_X)
                                data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), artifact_name="probabilities", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                                prediction = model.predict(test_classification_X)
                                data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                        elif split_type == "cart":
                            for criterion in ("gini", "entropy"):
                                base_model = decision_tree.DecisionTreeClassifier(split_type=split_type, criterion=criterion)
                                model = decision_tree.EnsembleLearning(base_estimator=base_model, method=method, n_estimators=n, max_samples=1.0, max_features=1.0, feature_selection=feature_selection, mask_keep_prob=0.7, oob_score=(method == "bagging"), random_state=None)

                                print(f"[LOG]: Ensemble Learning - n={n} - DecisionTreeClassifier - {split_type.upper()} - {base_model.criterion.upper()} - {method.upper()} - {feature_selection.upper()}")
                                model.fit(train_classification_X, train_classification_Y)
                                data_parser.save_run_artifact(OUT_DIR, np.array(model._feature_indices, dtype=object), get_params(model), artifact_name="feature_indices", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                                if method == "adaboost":
                                    data_parser.save_run_artifact(OUT_DIR, np.array(model._estimator_weights, dtype=float), get_params(model), artifact_name="estimator_weights", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                                data_parser.save_run_tree(OUT_DIR, [est._root for est in model._estimators], get_params(model), artifact_name="trees", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")

                                classification_metric._classes = np.unique(train_classification_Y)
                                classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
                                data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), artifact_name="probabilities", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                                prediction = model.predict(test_classification_X)
                                data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                        else:
                            base_model = decision_tree.DecisionTreeClassifier(split_type=split_type, criterion="gain_ratio")
                            model = decision_tree.EnsembleLearning(base_estimator=base_model, method=method, n_estimators=n, max_samples=1.0, max_features=1.0, feature_selection=feature_selection, mask_keep_prob=0.7, oob_score=(method == "bagging"), random_state=None)

                            print(f"[LOG]: Ensemble Learning - n={n} - DecisionTreeClassifier - {split_type.upper()} - {base_model.criterion.upper()} - {method.upper()} - {feature_selection.upper()}")
                            model.fit(train_classification_X, train_classification_Y)
                            data_parser.save_run_artifact(OUT_DIR, np.array(model._feature_indices, dtype=object), get_params(model), artifact_name="feature_indices", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                            if method == "adaboost":
                                data_parser.save_run_artifact(OUT_DIR, np.array(model._estimator_weights, dtype=float), get_params(model), artifact_name="estimator_weights", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                            data_parser.save_run_tree(OUT_DIR, [est._root for est in model._estimators], get_params(model), artifact_name="trees", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")

                            classification_metric._classes = np.unique(train_classification_Y)
                            classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
                            data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), artifact_name="probabilities", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                            prediction = model.predict(test_classification_X)
                            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_EnsembleLearning_n={n}_DecisionTreeClassifier_{base_model.criterion.upper()}_{method.title()}_{feature_selection.title()}_")
                elif base_model_name == "SVMClassifier":
                    kernels = ("linear", "rbf")

                    for kernel in kernels:
                        base_model = decision_tree.SVMClassifier(kernel=kernel)
                        model = decision_tree.EnsembleLearning(base_estimator=base_model, method=method, n_estimators=n, max_samples=1.0, max_features=1.0, feature_selection=feature_selection, mask_keep_prob=0.7, oob_score=(method == "bagging"), random_state=None)

                        print(f"[LOG]: Ensemble Learning - n={n} - SVMClassifier - {kernel.upper()} - {method.upper()} - {feature_selection.upper()}")
                        model.fit(train_classification_X, train_classification_Y)
                        data_parser.save_run_artifact(OUT_DIR, np.array(model._feature_indices, dtype=object), get_params(model), artifact_name="feature_indices", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")
                        if method == "adaboost":
                            data_parser.save_run_artifact(OUT_DIR, np.array(model._estimator_weights, dtype=float), get_params(model), artifact_name="estimator_weights", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")

                        if kernel == "linear":
                            data_parser.save_run_artifact(OUT_DIR, np.array([est._w for est in model._estimators], dtype=object), get_params(model), artifact_name="weights_list", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")
                            data_parser.save_run_artifact(OUT_DIR, np.array([est._b for est in model._estimators], dtype=float), get_params(model), artifact_name="bias_list", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")
                        else:
                            data_parser.save_run_artifact(OUT_DIR, np.array([est.alphas_ for est in model._estimators], dtype=object), get_params(model), artifact_name="alphas_list", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")
                            data_parser.save_run_artifact(OUT_DIR, np.array([est.b_ for est in model._estimators], dtype=float), get_params(model), artifact_name="bias_list", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")

                        classification_metric._classes = np.unique(train_classification_Y)
                        classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
                        data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), artifact_name="probabilities", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")
                        prediction = model.predict(test_classification_X)
                        data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, classification_metric.evaluate()), "prediction", run_id_prefix=f"run_EnsembleLearning_n={n}_SVMClassifier_{base_model.kernel.upper()}_{method.title()}_{feature_selection.title()}_")

# 3. NEURAL NETWORK

# Regression
hidden_layers = (1, 2, 3, 5, 10, 20, 30)
hidden_neurons = (1, 2, 3, 5, 10, 20, 30)
activation_functions = ("linear","tanh","sigmoid","relu","leaky_relu","special_relu")

for hidden_layer in hidden_layers:
    for hidden_neuron in hidden_neurons:
        for activation_function in activation_functions:
            model = neural_network.MLPRegressor(len(train_biased_regression_X[0]), hidden_layers=hidden_layer, hidden_size=hidden_neuron, activation=activation_function)
            
            print(f"[LOG]: Neural Network Regression - MLP Regression - layers={hidden_layer} - neurons={hidden_neuron} - {activation_function}")
            model.fit(train_biased_regression_X, train_regression_Y)
            data_parser.save_run_nn(OUT_DIR, model, get_params(model), artifact_name="neural_network", run_id_prefix=f"run_NeuralNetwork_MLPRegression_{hidden_layer}layers_{hidden_neuron}neurons_{activation_function}_")
            regression_metric.Y_pred = prediction = model.predict(test_biased_regression_X)
            data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), artifact_name="prediction", run_id_prefix=f"run_NeuralNetwork_MLPRegression_{hidden_layer}layers_{hidden_neuron}neurons_{activation_function}_")

# RBF Regression
n_values = (1, 2, 3, 5, 10, 20, 30)

for n in n_values:
    print(f"[LOG]: Neural Network Regression - RBF MLP Regressor - n-centers={n}")
    model = neural_network.RBFMLPRegressor(n_centers=n)

    model.fit(train_regression_X, train_regression_Y)
    data_parser.save_run_nn(OUT_DIR, model, get_params(model), artifact_name="neural_network", run_id_prefix=f"run_NeuralNetwork_RBFMLPRegression_{n}n-centers_")
    regression_metric.Y_pred = prediction = model.predict(test_regression_X)
    data_parser.save_run_artifact(OUT_DIR, prediction, get_params(model, regression_metric.evaluate()), artifact_name="prediction", run_id_prefix=f"run_NeuralNetwork_RBFMLPRegression_{n}n-centers_")

# Classification
hidden_layers = (1, 2, 3, 5, 10)
hidden_neurons = (1, 2, 3, 5, 10)
activation_functions = ("linear","tanh","sigmoid","relu","leaky_relu","special_relu")

for hidden_layer in hidden_layers:
    for hidden_neuron in hidden_neurons:
        for activation_function in activation_functions:
            model = neural_network.MLPClassifier(len(train_biased_classification_X[0]), hidden_layers=hidden_layer, hidden_size=hidden_neuron, activation=activation_function)
            
            print(f"[LOG]: Neural Network Classification - MLP Classifier - layers={hidden_layer} - neurons={hidden_neuron} - {activation_function}")
            model.fit(train_biased_classification_X, train_classification_Y)
            data_parser.save_run_nn(OUT_DIR, model, get_params(model), artifact_name="neural_network", run_id_prefix=f"run_NeuralNetwork_MLPClassification_{hidden_layer}layers_{hidden_neuron}neurons_{activation_function}_")
            classification_metric._classes = np.unique(train_classification_Y)
            classification_metric._Y_prob = prob = model.predict_proba(test_biased_classification_X)
            data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), artifact_name="probabilities", run_id_prefix=f"run_NeuralNetwork_MLPClassification_{hidden_layer}layers_{hidden_neuron}neurons_{activation_function}_")

# RBF Classifier
n_values = (1, 2, 3, 5, 10)

for n in n_values:
    print(f"[LOG]: Neural Network Classification - RBF MLP Classifier - n-centers={n}")
    model = neural_network.RBFMLPClassifier(n_centers=n)

    model.fit(train_classification_X, train_classification_Y)
    data_parser.save_run_nn(OUT_DIR, model, get_params(model), artifact_name="neural_network", run_id_prefix=f"run_NeuralNetwork_RBFMLPClassification_{n}n-centers_")
    classification_metric._classes = np.unique(train_classification_Y)
    classification_metric._Y_prob = prob = model.predict_proba(test_classification_X)
    data_parser.save_run_artifact(OUT_DIR, prob, get_params(model, classification_metric.evaluate()), artifact_name="probabilities", run_id_prefix=f"run_NeuralNetwork_RBFMLPClassification_{n}n-centers_")
