from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
            max_features: int = None,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.max_features: int = max_features    

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        """
        Обучает новую базовую модель и добавляет ее в ансамбль.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.
        y : array-like, форма (n_samples,)
            Массив целевых значений.
        predictions : array-like, форма (n_samples,)
            Предсказания текущего ансамбля.

        Примечания
        ----------
        Эта функция добавляет новую модель и обновляет ансамбль.
        """
        if isinstance(x, np.ndarray):
            idx = np.random.choice(len(x), int(len(x) * self.subsample), replace=False)
            x_subsample, y_subsample = x[idx], y[idx]
        else:
            # Если x - разреженный массив
            idx = np.random.choice(x.shape[0], int(x.shape[0] * self.subsample), replace=False)
            x_subsample = x[idx]
            y_subsample = y[idx]
        new_model = self.base_model_class(**{**self.base_model_params, 'max_features': self.max_features})
#         new_model = self.base_model_class(**self.base_model_params)
        new_model.fit(x_subsample, y_subsample - predictions[idx])

        gamma = self.find_optimal_gamma(y_subsample, predictions[idx], new_model.predict(x_subsample))
        self.gammas.append(gamma * self.learning_rate)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.

        Параметры
        ----------
        x_train : array-like, форма (n_samples, n_features)
            Массив признаков для тренировочного набора.
        y_train : array-like, форма (n_samples,)
            Массив целевых значений для тренировочного набора.
        x_valid : array-like, форма (n_samples, n_features)
            Массив признаков для валидационного набора.
        y_valid : array-like, форма (n_samples,)
            Массив целевых значений для валидационного набора.
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            if self.early_stopping_rounds is not None:
                valid_predictions += self.learning_rate * self.models[-1].predict(x_valid)
                train_predictions += self.learning_rate * self.models[-1].predict(x_train)

                train_loss = self.loss_fn(y_train, train_predictions)
                valid_loss = self.loss_fn(y_valid, valid_predictions)

                self.history['train_loss'].append(train_loss)
                self.history['valid_loss'].append(valid_loss)
                self.history['train_score'].append(score(self, x_train, y_train))
                self.history['valid_score'].append(score(self, x_valid, y_valid))

                if len(self.validation_loss) > 0 and valid_loss < min(self.validation_loss):
                    self.validation_loss[:-1] = self.validation_loss[1:]
                    self.validation_loss[-1] = valid_loss
                else:
                    self.validation_loss[:-1] = self.validation_loss[1:]
                    self.validation_loss[-1] = valid_loss

                if np.all(self.validation_loss == self.validation_loss[0]):
                    break

        if self.plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(self.history['train_loss'], label='Train Loss')
            ax[0].plot(self.history['valid_loss'], label='Valid Loss')
            ax[0].set_title('Loss')
            ax[0].legend()

            ax[1].plot(self.history['train_score'], label='Train Score')
            ax[1].plot(self.history['valid_score'], label='Valid Score')
            ax[1].set_title('Score')
            ax[1].legend()
            plt.show()

    def predict_proba(self, x):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.

        Возвращает
        ----------
        probabilities : array-like, форма (n_samples, n_classes)
            Вероятности для каждого класса.
        """
        probabilities = np.zeros((x.shape[0], 2))
        for gamma, model in zip(self.gammas, self.models):
            probabilities[:, 1] += gamma * model.predict(x)
        probabilities[:, 0] = 1 - probabilities[:, 1]
        
#         probabilities = np.clip(probabilities, 0, 1)  # Ограничение вероятностей диапазоном [0, 1] (без него - скор выше)
        return probabilities
    
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.

        Примечания
        ----------
        Значение гаммы определяется путем минимизации функции потерь.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        importances = np.zeros(self.max_features)
        for model in self.models:
            importances += self.learning_rate * model.feature_importances_
        importances /= importances.sum()  # Нормализация
        return importances
