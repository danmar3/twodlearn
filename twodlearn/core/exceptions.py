
class PropertyRedefinition(Exception):
    # Constructor or Initializer
    def __init__(self, property, object):
        self.property = property
        self.object = object

    def __str__(self):
        return ('The property {} in object {} has already been set'
                ''.format(repr(self.property), repr(self.object)))


class UnsetProperty(Exception):
    # Constructor or Initializer
    def __init__(self, property, object):
        self.property = property
        self.object = object

    def __str__(self):
        return ('The property {} in object {} has not been set'
                ''.format(repr(self.property), repr(self.object)))


class InitPreconditionsFailed(Exception):
    # Constructor or Initializer
    def __init__(self, object, property, reqs=None):
        self.property = property
        self.object = object
        self.reqs = reqs

    def __str__(self):
        msg = ('The preconditions for initializing {} in object {} '
               'failed.'.format(repr(self.property), repr(self.object)))
        if self.reqs is not None:
            msg = msg + ' Initialization requires any of: {}'.format(self.reqs)
        return msg


class AutoInitFailed(Exception):
    # Constructor or Initializer
    def __init__(self, property, object):
        self.property = property
        self.object = object

    def __str__(self):
        return ('Auto initialization of property {} in object {} failed'
                ''.format(repr(self.property), repr(self.object)))


class NonePropertyAvailable(Exception):
    def __init__(self, object, property=None, reqs=None):
        self.object = object
        self.reqs = reqs
        self.property = property

    def __str__(self):
        if self.property is None:
            return ('Object {} has not been provided with any of the required '
                    'properties {}'.format(self.object, self.reqs))
        else:
            return ('Initialization of {}.{} requires any of {} to be provided'
                    ''.format(self.object, self.property, self.reqs))
