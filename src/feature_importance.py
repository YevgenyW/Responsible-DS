import pandas as pd
import altair as alt

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

def analyze(model_prediction, X_train, render=False):
    skater_model = InMemoryModel(model_prediction, examples=X_train)
    interpreter = Interpretation(X_train, feature_names=X_train.columns)
    
    result = interpreter.feature_importance.feature_importance(skater_model, ascending=False)
    
    if render:
        return render_feature_importance(result)
    else:
        return result

def render_feature_importance(result):
    result = pd.DataFrame({'feature': result.index, 'importance': result.values})
    
    return alt.Chart(result).mark_bar().encode(
        x=alt.X('importance', axis=alt.Axis(title='')),
        y=alt.Y(
            'feature',
            axis=alt.Axis(title=''),
            sort=alt.EncodingSortField(field='importance', op='sum', order='descending')
        )
    )
    