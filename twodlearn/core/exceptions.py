'''Definition of tdl specific errors and exceptions
'''

import inspect


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
    def __init__(self, property, object, msg=None):
        self.property = property
        self.object = object
        self.msg = msg

    def __str__(self):
        err_msg = ('Auto initialization of property {} in object {} failed'
                   ''.format(repr(self.property), repr(self.object)))
        if self.msg is None:
            return err_msg
        else:
            return err_msg + self.msg


class ArgumentNotProvided(Exception):
    def __init__(self, object, property=None):
        self.object = object
        if property is None:
            self.property = str(inspect.stack()[1][3])  # .function)

    def __str__(self):
        return ("{} missing 1 required argument: '{}' "
                "".format(repr(self.object), repr(self.property)))


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


class NanResult(Exception):
    pass
