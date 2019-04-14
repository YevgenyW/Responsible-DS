from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

def get_features(model_prediction, X_test):
	skater_model = InMemoryModel(model_prediction, examples=X_test)
	interpreter = Interpretation(X_test, feature_names=X_test.columns)
	return interpreter.feature_importance.feature_importance(skater_model, ascending=False)
	# feature_importance

def get_features_2(model_prediction_simple, X_train, model_classes):
	skater_model = InMemoryModel(model_prediction_simple, examples=X_train, unique_values=model_classes)
	interpreter = Interpretation(X_train, feature_names=X_train.columns)
	return interpreter.feature_importance.feature_importance(skater_model, ascending=False)