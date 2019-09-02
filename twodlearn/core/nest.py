from tensorflow import __version__ as _tf_version
try:
    from tensorflow import nest as _tf_nest
except ImportError:
    from tensorflow.contrib.framework import nest as _tf_nest


assert_same_structure = _tf_nest.assert_same_structure
pack_sequence_as = _tf_nest.pack_sequence_as
map_structure = _tf_nest.map_structure
flatten = _tf_nest.flatten

if [int(s) for s in _tf_version.split('.')[0:2]] < [1, 14]:
    is_nested = _tf_nest.is_sequence
else:
    is_nested = _tf_nest.is_nested
