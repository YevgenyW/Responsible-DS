import pandas as pd
import numpy as np
import sklearn
import warnings
from sklearn.utils import check_random_state
from .rules import Literal, Operator
from .utils import cache, check_stringvar, show_image, softmax, rbf

# Suppress FutureWarning of sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)


class DomainMapper:
    def __init__(self,
                 train_data,
                 contrast_names=None,
                 kernel_width=.25,
                 seed=1):
        '''Init.

        Args:
            train_data: Original data, to obtain input distributions
                for generating neighborhood data
            contrast_names (list/dict): Names of contrasts (including fact
                and foil)
            seed (int): Seed for random functions
        '''
        self.train_data = train_data
        self.kernel_fn = lambda d: rbf(d, sigma=kernel_width)

        if type(contrast_names) is list:
            contrast_names = dict(enumerate(contrast_names))
        self.contrast_class = contrast_names
        self.contrast_map = None
        if type(self.contrast_class) is dict:
            self.contrast_map = [*self.contrast_class]

        self.seed_ = seed
        self.seed = check_random_state(seed)

    def map_contrast_names(self,
                           contrast):
        '''Map a descriptive name to a contrast if present.

        Args:
            contrast (int): Identifier of contrast
        '''
        # print("contrast = ", contrast)
        if self.contrast_class is not None:
            if self.contrast_map is not None:
                # print("self.contrast_class = ", self.contrast_class)
                return self.contrast_class[self.contrast_map[contrast]]
            return self.contrast_class[contrast]
        return contrast

    def _weights(self,
                 data,
                 distance_metric,
                 sample=None):
        '''Calculate sample weights based on distance metric.'''
        if sample is None:
            sample = data[0].reshape(1, -1)
        distances = sklearn.metrics.pairwise_distances(
            data,
            sample,
            metric=distance_metric
        ).ravel()
        return self.kernel_fn(distances)

    def _data(self,
              data,
              scaled_data,
              distance_metric,
              predict_fn):
        # Calculate weights
        if scaled_data is None:
            scaled_data = data
        weights = self._weights(scaled_data, distance_metric)

        # Predict; distinguish between .predict and .predict_proba
        preds = predict_fn(data)
        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)

        return data, weights, preds

    def unweighted_sample_training(self,
                                   predict_fn,
                                   n_samples,
                                   seed=1,
                                   **kwargs):
        if self.train_data is None:
            raise Exception('Can only sample from training data '
                            'when it is made available')

        # Predict
        ys_p = predict_fn(self.train_data)
        ys = ys_p.argmax(axis=1) if ys_p.ndim > 1 else ys_p

        # TODO: make prob based on class distrib
        self.seed = check_random_state(seed)
        to_select = self.seed.choice(range(len(ys)),
                                     size=n_samples)

        return self.train_data[to_select], ys[to_select], ys_p[to_select]

    @cache
    def sample_training_data(self,
                             sample,
                             predict_fn,
                             distance_metric='euclidean',
                             n_samples=500,
                             seed=1,
                             **kwargs):
        xs, ys, _ = self.unweighted_sample_training(predict_fn,
                                                    n_samples=n_samples,
                                                    seed=seed,
                                                    **kwargs)
        return xs, self._weights(xs, distance_metric), ys, sample

    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=500,
                                   **kwargs):
        raise NotImplementedError('Implemented in subclasses')

    def map_feature_names(self, descriptive_path, remove_last=False):
        raise NotImplementedError('Implemented in subclasses')

    def explain(self, fact, foil, counterfactuals, confidence, **kwargs):
        raise NotImplementedError('Implemented in subclasses')


class DomainMapperTabular(DomainMapper):
    '''Domain mapper for tabular data (columns and rows,
    with feature names for columns).'''

    def __init__(self,
                 train_data,
                 feature_names,
                 contrast_names=None,
                 categorical_features=None,
                 kernel_width=None,
                 seed=1):
        '''Init.

        Args:
            feature_names (list): Feature names (should be same length
                as # columns)
            categorical_features (list): Indices of categorical features
            kernel_width: Kernel width for deciding on weights of data
        '''
        if kernel_width is None:
            kernel_width = np.sqrt(train_data.shape[1]) * .75

        super().__init__(train_data=train_data,
                         contrast_names=contrast_names,
                         kernel_width=kernel_width,
                         seed=seed)

        if type(train_data) is pd.core.frame.DataFrame:
            train_data = train_data.as_matrix()
        self.train_data = train_data
        if feature_names is None:
            feature_names = [i for i in range(train_data.shape[1])]
        self.features = feature_names
        self.categorical_features = categorical_features

    @cache
    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=500,
                                   seed=1,
                                   **kwargs):
        '''Generate neighborhood data for a given point (currently using LIME)

        Args:
            train_data: Training data predict_fn was trained on
            sample: Observed sample
            predict_fn: Black box predictor to predict all points
            distance_metric: Distance metric used for weights
            n_samples: Number of samples to generate

        Returns:
            neighor_data (xs around sample),
            weights (weights of instances in xs),
            neighor_data_labels (ys around sample, corresponding to xs)
        '''
        from lime.lime_tabular import LimeTabularExplainer
        e = LimeTabularExplainer(self.train_data,
                                 categorical_features=self.categorical_features,
                                 discretize_continuous=False)

        _, neighbor_data = e._LimeTabularExplainer__data_inverse(sample,
                                                                 n_samples)
        scaled_data = (neighbor_data - e.scaler.mean_) / e.scaler.scale_
        return (*self._data(neighbor_data, scaled_data,
                            distance_metric, predict_fn),
                sample)

    def map_feature_names(self, explanation, remove_last=False):
        '''Replace feature ids with feature names in a descriptive path.

        Args:
            explanation: Explanation obtained with
                get_explanation() or descriptive_path() function
            feature_names: Feature names
            remove_last: Remove last tuple from explanation

        Returns:
            Explanation with feature names mapped
        '''
        def get_feature(x):
            ret = x
            if x[0] >= 0 and x[0] < len(self.features):
                ret = list(x)
                ret[0] = self.features[ret[0]]
                if type(x) is Literal:
                    ret = Literal(*ret)
                else:
                    ret = type(x)(ret)
            if (type(ret) is Literal and self.categorical_features is not None):
                if x[0] in self.categorical_features:
                    ret.categorical = True
            return ret

        if self.features is not None:
            ex = [get_feature(e) for e in explanation]

            if remove_last:
                ex = ex[:-1]
        return ex

    def rule_to_str(self,
                    rule,
                    remove_last=False):
        rule = rule or []
        return ' and '.join([str(c) for c in self.map_feature_names(rule, 
                               remove_last=remove_last)])

    def explain(self,
                fact,
                foil,
                counterfactuals,
                factuals,
                confidence,
                fidelity,
                time,
                **kwargs):
        '''Explain an instance using the results of
        ContrastiveExplanation.explain_instance()

        Args:
            fact: ID of fact
            foil: ID of foil
            counterfactuals: List of Literals that form
                explanation as disjoint set of foil and
                not-fact rules
            factuals: List of Literals that form explanation
                as set of fact rules
            confidence [0, 1]: Confidence of explanation
                on neighborhood data
            fidelity: ...
            time: Time taken to explain (s)
        '''

        fact = self.map_contrast_names(fact)
        foil = self.map_contrast_names(foil)

        e = f"The model predicted '{fact}' instead of '{foil}' " \
            f"because '{self.rule_to_str(counterfactuals)}'"

        if factuals is None:
            return e
        else:
            return (e, f"The model predicted '{fact}' because "
                       f"'{self.rule_to_str(factuals, remove_last=True)}'")


class DomainMapperImage(DomainMapper):
    '''Domain mapper for image data (CNN-only) using feature_fn.'''

    def __init__(self,
                 train_data,
                 feature_fn,
                 contrast_names=None,
                 kernel_width=.25,
                 seed=1):
        super().__init__(train_data, contrast_names=contrast_names,
                         kernel_width=kernel_width, seed=seed)
        self.feature_fn = feature_fn

    @cache
    def sample_training_data(self,
                             sample,
                             predict_fn,
                             distance_metric='euclidean',
                             n_samples=100,
                             seed=1,
                             **kwargs):
        xs, ys, _ = super().unweighted_sample_training(predict_fn,
                                                       n_samples=n_samples,
                                                       seed=seed,
                                                       **kwargs)
        xs = self.feature_fn(xs)
        return (xs, self._weights(xs, distance_metric),
                ys, self.feature_fn(sample))

    @cache
    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=100,
                                   seed=1,
                                   **kwargs):
        raise NotImplementedError()

    def map_feature_names(self, descriptive_path, remove_last=False):
        raise NotImplementedError()

    def explain(self, fact, foil, counterfactuals, factuals, confidence, fidelity, time):
        print("!2")
        fact = self.map_contrast_names(fact)
        foil = self.map_contrast_names(foil)

        return (f"The model predicted '{fact}' instead of '{foil}'",
                counterfactuals)


class DomainMapperImageSegments(DomainMapper):
    '''Domain mapper for image data (model-agnostic) using segments.'''

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.image = None
        self.segments = None
        self.alt_image = None

    def n_features(self):
        if self.segments is None:
            return 0
        return np.unique(self.segments).shape[0]

    def data_labels(self,
                    predict_fn,
                    batch_size=30):
        from copy import deepcopy
        from itertools import product
        data = np.array([list(i) for i in product([0, 1],
                         repeat=self.n_features())])
        labels = []
        imgs = []
        for row in data:
            temp = deepcopy(self.image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(self.segments.shape).astype(bool)
            for z in zeros:
                mask[self.segments == z] = True
            temp[mask] = self.alt_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = predict_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = predict_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.argmax(np.array(labels), axis=1)

    @check_stringvar(('segmentation_fn', ['quickshift', 'felzenszwalb', 'slic']))
    def __high_level_features(self,
                              sample,
                              predict_fn,
                              distance_metric='euclidean',
                              segmentation_fn='quickshift',
                              alt_image=None,
                              hide_color=None,
                              n_samples=30,
                              batch_size=30,
                              seed=1):
        '''...'''
        from skimage.segmentation import quickshift, felzenszwalb, slic

        self.image = sample

        # Segment
        if segmentation_fn == 'felzenschwalb':
            segments = felzenszwalb(sample)
        elif segmentation_fn == 'slic':
            segments = slic(sample, n_segments=10)
        else:
            segments = quickshift(sample, kernel_size=2,
                                  max_dist=200, ratio=0.2,
                                  random_seed=seed)
        self.segments = segments

        # Get fudged image, unless given
        if alt_image is None:
            alt_image = sample.copy()
            if hide_color is None:
                for x in np.unique(segments):
                    alt_image[segments == x] = (
                        np.mean(sample[segments == x][:, 0]),
                        np.mean(sample[segments == x][:, 1]),
                        np.mean(sample[segments == x][:, 2]))
            else:
                alt_image[:] = hide_color
        self.alt_image = alt_image

        data, preds = self.data_labels(predict_fn, batch_size=batch_size)

        return (data, self._weights(data, distance_metric),
                preds, np.zeros(self.n_features()))

    @cache
    def sample_training_data(self,
                             sample,
                             predict_fn,
                             distance_metric='euclidean',
                             n_samples=30,
                             seed=1,
                             foil_encode_fn=None,
                             **kwargs):
        fn = super().unweighted_sample_training
        if foil_encode_fn is None:
            self.alt_image = fn(predict_fn, n_samples=1, seed=seed,
                                **kwargs)[0][0]
        else:  # Foil-sensitive
            data, preds, preds_probs = fn(predict_fn, n_samples=n_samples,
                                          seed=seed, **kwargs)

            # Highest confidence foil
            foils = np.argwhere(foil_encode_fn(preds) == 1).ravel()
            foils_p = [preds_probs[f][preds[f]] for f in foils]
            self.alt_image = data[foils[np.argmax(foils_p)]]

        return self.__high_level_features(sample,
                                          predict_fn,
                                          distance_metric=distance_metric,
                                          alt_image=self.alt_image,
                                          n_samples=n_samples,
                                          seed=seed)

    @cache
    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=30,
                                   batch_size=10,
                                   seed=1,
                                   hide_color=None,
                                   segmentation_fn='quickshift',
                                   **kwargs):
        while sample.ndim > 3:
            sample = sample[0]

        return self.__high_level_features(sample,
                                          predict_fn,
                                          distance_metric=distance_metric,
                                          segmentation_fn=segmentation_fn,
                                          hide_color=hide_color,
                                          n_samples=n_samples,
                                          batch_size=batch_size,
                                          seed=seed)

    def map_feature_names(self, explanation):
        if (self.image is None or self.alt_image is None or
                self.segments is None):
            return

        for e in explanation:
            if type(e) is Literal:
                temp = np.zeros(self.image.shape)
                temp[self.segments == e.feature] = 1
                if e.operator is Operator.SEQ:
                    temp = self.image * temp
                elif e.operator is Operator.GT:
                    temp = self.alt_image * temp
                show_image(temp)
        return explanation

    def explain(self, fact, foil, counterfactuals, factuals,
                confidence, fidelity, time, **kwargs):
        print("!1")
        fact = self.map_contrast_names(fact)
        foil = self.map_contrast_names(foil)

        print(fact)
        show_image(self.image)
        print(foil)
        show_image(self.alt_image)

        self.map_feature_names(counterfactuals)

        return (f"The model predicted '{fact}' instead of '{foil}'",
                counterfactuals)
