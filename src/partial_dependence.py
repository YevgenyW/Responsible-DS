import altair as alt

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

def analyze(features, model_prediction, X_test, resolution=20, render=False):
    skater_model = InMemoryModel(model_prediction, examples=X_test)
    interpreter = Interpretation(X_test, feature_names=X_test.columns)
    
    result = interpreter.partial_dependence.partial_dependence(features, skater_model, grid_resolution=resolution)
    result.rename(columns={1: 'Prediction'}, inplace=True)

    if render:
        return render_partial_dependence(result, features)
    else:
        return result

def render_partial_dependence(result, features):   
    error_msg = '1 to 2 features should be specified'
    
    if len(features) == 0:
        raise Exception(error_msg)
    elif len(features) == 1:
        return alt.Chart(result).mark_line().encode(
            x=alt.X(f'{features[0]}:O', axis=alt.Axis(format='.1f')),
            y='Prediction:Q'
        )
    elif len(features) == 2:
        return alt.Chart(result).mark_rect().encode(
            x=alt.X(f'{features[0]}:O', axis=alt.Axis(format='.1f')),
            y=alt.Y(f'{features[1]}:O', axis=alt.Axis(format='.1f')),
            color='Prediction:Q',
            tooltip='Prediction:Q'
        )
    else:
        raise Exception(error_msg)
        