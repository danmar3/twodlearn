
class _Shortcut(object):
    def __init__(self, namespace, attr):
        self.namespace = namespace
        self.attr = attr

    def __call__(self, obj):
        return getattr(getattr(obj, self.namespace), self.attr)


class PropertyShortcuts(object):
    ''' add a set of property shortcuts to a class '''
    def __init__(self, shortcuts):
        if not isinstance(shortcuts, dict):
            raise ValueError('Shortcuts should be a dictionary with '
                             '(namespace, [attrs]) as (key, value) pairs.')
        attrs = set()
        for namespace in shortcuts:
            if any(attr_i in attrs for attr_i in shortcuts[namespace]):
                raise ValueError('Found duplicated attributes in shortcut '
                                 'dictionary {}'.format(shortcuts))
        self.shortcuts = shortcuts

    def __call__(self, cls):
        for namespace in self.shortcuts.keys():
            for attr in self.shortcuts[namespace]:
                setattr(cls, attr, property(_Shortcut(namespace, attr)))
        return cls
