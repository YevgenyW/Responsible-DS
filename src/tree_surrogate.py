from IPython.display import Image

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

def analyze(model_prediction, X_train, y_train):
    skater_model = InMemoryModel(model_prediction, examples=X_train)
    interpreter = Interpretation(X_train, feature_names=X_train.columns)
    
    surrogate_explainer = interpreter.tree_surrogate(skater_model, seed=5)
    surrogate_explainer.fit(X_train, y_train, use_oracle=True, prune='post', scorer_type='default')
    surrogate_explainer.plot_global_decisions(colors=['coral', 'lightsteelblue','darkkhaki'],
                                              file_name='simple_tree_pre.png')
    
    return Image(filename='simple_tree_pre.png')
