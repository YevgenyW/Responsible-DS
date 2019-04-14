from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

def get_dependency(model_prediction, X_test, args, resolution = 30):
	skater_model = InMemoryModel(model_prediction, examples=X_test)
	interpreter = Interpretation(X_test, feature_names=X_test.columns)
	return interpreter.partial_dependence.partial_dependence(args, skater_model, grid_resolution=resolution)

def get_dependency_2(model_prediction, X_test):
	skater_model = InMemoryModel(model_prediction, examples=X_test)
	interpreter = Interpretation(X_test, feature_names=X_test.columns)
	return interpreter.partial_dependence.partial_dependence(['ApplicantIncome'], skater_model, grid_resolution=30)