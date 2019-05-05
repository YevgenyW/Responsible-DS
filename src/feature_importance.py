import pandas as pd
import altair as alt

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

def analyze(model_prediction, X_test, render=False):
    skater_model = InMemoryModel(model_prediction, examples=X_test)
    interpreter = Interpretation(X_test, feature_names=X_test.columns)
    
    result = interpreter.feature_importance.feature_importance(skater_model, ascending=False)
    result = pd.DataFrame({'feature': result.index, 'importance': result.values})
    
    if render:
        return render_feature_importance(result)
    else:
        return result

def render_feature_importance(result):
    return alt.Chart(result).mark_bar().encode(
        x=alt.X('importance', axis=alt.Axis(title='')),
        y=alt.Y(
            'feature',
            axis=alt.Axis(title=''),
            sort=alt.EncodingSortField(field='importance', op='sum', order='descending')
        )
    )
    