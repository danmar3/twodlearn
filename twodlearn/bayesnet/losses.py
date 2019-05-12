from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import twodlearn as tdl
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False

_KL_DIVERGENCES = {}


class KLDivergence(tdl.core.TdlModel):
    ''' Get the KL-divergence KL(distribution_a || distribution_b).
    If there is no KL method registered specifically for `type(distribution_a)`
    and `type(distribution_b)`, then the class hierarchies of these types are
    searched.

    If one KL method is registered between any pairs of classes in these two
    parent hierarchies, it is used.

    If more than one such registered method exists, the method whose registered
    classes have the shortest sum MRO paths to the input types is used.

    If more than one such shortest path exists, the first method
    identified in the search is used (favoring a shorter MRO distance to
    `type(distribution_a)`).
    '''
    @staticmethod
    def _registered_kl(type_a, type_b):
        """Get the KL function registered for classes a and b."""
        hierarchy_a = inspect.getmro(type_a)
        hierarchy_b = inspect.getmro(type_b)
        dist_to_children = None
        kl_fn = None
        for mro_to_a, parent_a in enumerate(hierarchy_a):
            for mro_to_b, parent_b in enumerate(hierarchy_b):
                candidate_dist = mro_to_a + mro_to_b
                candidate_kl_fn = _KL_DIVERGENCES.get((parent_a, parent_b),
                                                      None)
                if not kl_fn or (candidate_kl_fn and
                                 candidate_dist < dist_to_children):
                    dist_to_children = candidate_dist
                    kl_fn = candidate_kl_fn
        return kl_fn

    @tdl.core.InputModel
    def distribution_a(self, value):
        return value

    @tdl.core.InputModel
    def distribution_b(self, value):
        return value

    @tdl.core.OutputValue
    def value(self, _):
        tdl.core.assert_initialized(self, 'value', ['distribution_a',
                                                    'distribution_b'])
        distribution_a = self.distribution_a
        distribution_b = self.distribution_b
        kl_fn = self._registered_kl(type(self.distribution_a),
                                    type(self.distribution_b))
        if kl_fn is None and not TFP_AVAILABLE:
            raise NotImplementedError(
                    "No KL(distribution_a || distribution_b) registered for "
                    "distribution_a type %s and distribution_b type %s"
                    % (type(distribution_a).__name__,
                       type(distribution_b).__name__))
        if kl_fn is None and TFP_AVAILABLE:
            kl_fn = tfp.distributions.kl_divergence
        value = kl_fn(distribution_a, distribution_b)
        return value

    def __init__(self, distribution_a, distribution_b, name=None):
        super(KLDivergence, self).__init__(
            distribution_a=distribution_a,
            distribution_b=distribution_b,
            name=name)


class RegisterKL(object):
    """Decorator to register a KL divergence implementation function.
    Usage:
    @distributions.RegisterKL(distributions.Normal, distributions.Normal)
    def _kl_normal_mvn(norm_a, norm_b):
        # Return KL(norm_a || norm_b)
    """
    def __init__(self, dist_cls_a, dist_cls_b):
        """Initialize the KL registrar.
        Args:
          dist_cls_a: the class of the first argument of the KL divergence.
          dist_cls_b: the class of the second argument of the KL divergence.
        """
        self._key = (dist_cls_a, dist_cls_b)

    def __call__(self, kl_fn):
        """Perform the KL registration.
        Args:
          kl_fn: The function to use for the KL divergence.
        Returns:
          kl_fn
        Raises:
          TypeError: if kl_fn is not a callable.
          ValueError: if a KL divergence function has already been registered
            for the given argument classes.
        """
        if not callable(kl_fn):
            raise TypeError("kl_fn must be callable, received: %s" % kl_fn)
        if self._key in _KL_DIVERGENCES:
            raise ValueError("KL(%s || %s) has already been registered to: %s"
                             % (self._key[0].__name__, self._key[1].__name__,
                                _KL_DIVERGENCES[self._key]))
        _KL_DIVERGENCES[self._key] = kl_fn
        return kl_fn
