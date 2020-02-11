import numpy as np
import src.util as util
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


class Classifier():
    def __init__(self, model=None, model_file_path=None):
        self.model = model

        if model_file_path is not None:
            self.load(model_file_path)


    def train(self, X_training, X_validation, y_training, y_validation, class_names=[0, 1], verbose=True):
        self.model.fit(X_training, y_training)

        y_validation_pred = self.model.predict(X_validation)

        precision_validation = precision_score(y_validation, y_validation_pred, average="macro")
        recall_validation = recall_score(y_validation, y_validation_pred, average="macro")
        f1_validation = f1_score(y_validation, y_validation_pred, average="macro")
        c_matrix_validation = confusion_matrix(y_validation, y_validation_pred, class_names)

        if verbose:
            #print('============================================')
            print('Quality on validation dataset: precision = %f, recall = %f, f1 = %f' % (precision_validation, recall_validation, f1_validation))
            util.plot_confusion_matrix(c_matrix_validation, classes=class_names)

        score_dict = {'precision': precision_validation, 'recall': recall_validation,
                      'f1': f1_validation, 'confusion_matrix': c_matrix_validation}

        return score_dict


    def cross_validate(self, num_folds, X_cross_validation, y_cross_validation, class_names=[0, 1], seed=None, verbose=True):
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

        i = 0
        cv_score_dict = {'precision': [], 'recall': [], 'f1': [], 'confusion_matrix': []}

        for training_indexes, validation_indexes in kfold.split(X_cross_validation, y_cross_validation):
            X_training = X_cross_validation.iloc[training_indexes]
            y_training = y_cross_validation.iloc[training_indexes]

            X_validation = X_cross_validation.iloc[validation_indexes]
            y_validation = y_cross_validation.iloc[validation_indexes]

            self.model.fit(X_training, y_training)

            y_validation_pred = self.model.predict(X_validation)

            precision_validation = precision_score(y_validation, y_validation_pred, average="macro")
            recall_validation = recall_score(y_validation, y_validation_pred, average="macro")
            f1_validation = f1_score(y_validation, y_validation_pred, average="macro")
            c_matrix_validation = confusion_matrix(y_validation, y_validation_pred)

            cv_score_dict['precision'].append(precision_validation)
            cv_score_dict['recall'].append(recall_validation)
            cv_score_dict['f1'].append(f1_validation)
            cv_score_dict['confusion_matrix'].append(c_matrix_validation)

            if verbose:
                print('============================================')
                print('Fold %d quality: precision = %f, recall = %f, f1 = %f' % (i + 1, precision_validation, recall_validation, f1_validation))
                util.plot_confusion_matrix(c_matrix_validation, classes=class_names)

            i += 1

        if verbose:
            print('============================================')
            print('Mean quality of cross-validation: precision = %f, recall = %f, f1 = %f' % (np.mean(cv_score_dict['precision']), np.mean(cv_score_dict['recall']), np.mean(cv_score_dict['f1'])))

        return cv_score_dict

    def evaluate(self, X_test, y_test, class_names=[0, 1], verbose=True):
        y_test_pred = self.model.predict(X_test)

        precision_test = precision_score(y_test, y_test_pred, average="macro")
        recall_test = recall_score(y_test, y_test_pred, average="macro")
        f1_test = f1_score(y_test, y_test_pred, average="macro")
        c_matrix_test = confusion_matrix(y_test, y_test_pred)

        if verbose:
            print('============================================')
            print('Quality on evaluation dataset: precision = %f, recall = %f, f1 = %f' % (precision_test, recall_test, f1_test))
            util.plot_confusion_matrix(c_matrix_test, classes=class_names)

        score_dict = {'precision': precision_test, 'recall': recall_test,
                      'f1': f1_test, 'confusion_matrix': c_matrix_test}

        return score_dict


    def predict(self, X):
        y_pred = self.model.predict(X)

        return y_pred


    def predict_instance(self, instance):
        return self.model.predict(instance)


    def load(self, file_path):
        with open(file_path, "rb") as file:
            self.model = pickle.load(file)


    def save(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)
