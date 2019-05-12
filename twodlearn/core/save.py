import collections
import tensorflow as tf
import twodlearn.core.common


DATA_MODELS = set()


class register_saver(object):
    def __init__(self, ModelClass, ModelData):
        DATA_MODELS.add(ModelData)
        self.ModelClass = ModelClass

    def __call__(self, save_func):
        self.ModelClass.get_save_data = save_func


def load_data(data):
    known_data = set([ModelData, ListModelData])
    if issubclass(type(data), tuple(set.union(DATA_MODELS, known_data))):
        return data.load()
    else:
        return data


def load_variables(model, data):
    if data is None:
        return
    saved_vars = {var.init_args['name']: var.init_args['initial_value']
                  for var in data}
    model_vars = twodlearn.core.common.get_variables(model)
    for var in model_vars:
        var_name = var.name.split(':')[0]
        if var_name in saved_vars:
            data = saved_vars[var_name]
            with tf.name_scope(var_name):
                x = tf.placeholder(dtype=var.dtype.base_dtype)
                tf.assign(var, x).eval(feed_dict={x: data})


class ListModelData(object):
    def __init__(self, items):
        self.items = items

    @twodlearn.core.common.LazzyProperty
    def _loaded(self):
        items = [load_data(value) for value in self.items]
        return items

    def load(self):
        return self._loaded


class ModelData(object):
    def __init__(self, cls, init_args=None, variables=None):
        self.cls = cls
        self.init_args = (init_args if init_args is not None
                          else dict())
        self.variables = variables

    def init(self):
        if self.cls is None:
            return None
        init_args = {k: v.init() for k, v in self.init_args.items()}
        return self.cls(**init_args)

    @twodlearn.core.common.LazzyProperty
    def _loaded(self):
        if self.cls is None:
            return None
        elif self.cls == tf.Variable:
            return tf.Variable(**self.init_args)
            # init_args = self.init_args
            # return tf.get_variable(**init_args)
        elif issubclass(self.cls, tf.Tensor):
            return tf.convert_to_tensor(self.init_args['data'])
        elif issubclass(self.cls, twodlearn.core.common.TdlModel):
            inputs = {key: load_data(value)
                      for key, value in self.init_args.items()}
            model = self.cls(**inputs)
            load_variables(model, data=self.variables)
            return model
        else:
            return self.init_args['data']

    def load(self):
        return self._loaded

    def __str__(self):
        return ("<class:{}, init_args: {}> \n"
                "".format(self.cls, self.init_args.keys()))

    def __repr__(self):
        return ("<class:{}, init_args: {}> \n"
                "".format(self.cls, self.init_args.keys()))


SAVED_DATA = dict()
SAVER_COUNTER = 0


def get_save_data(model, include_variables=True):
    global SAVER_COUNTER
    SAVER_COUNTER += 1

    if model is None:
        SAVER_COUNTER -= 1
        return ModelData(None)
    if isinstance(model, collections.Hashable) and model in SAVED_DATA:
        SAVER_COUNTER -= 1
        return SAVED_DATA[model]
    elif isinstance(model, list):
        data = ListModelData(items=[get_save_data(v) for v in model])
    elif type(model) == tf.Variable:
        dtype = model.dtype.base_dtype
        data = ModelData(
            cls=tf.Variable,
            init_args={'initial_value': model.eval(),
                       # 'initializer': model.eval(),
                       'trainable': twodlearn.core.common.is_trainable(model),
                       'dtype': dtype,
                       'name': model.name.split(':')[0]}
            )
    elif isinstance(model, tf.Tensor):
        data = ModelData(cls=tf.Tensor,
                         init_args={'data': model.eval()})
    elif isinstance(model, twodlearn.core.common.TdlModel):
        if hasattr(model, 'get_save_data'):
            data = model.get_save_data()
        else:
            input_args = model.__tdl__._input_args
            init_args = {arg: get_save_data(getattr(model, arg))
                         for arg in input_args}
            init_args.update({'name': model.scope})

            data = ModelData(cls=type(model),
                             init_args=init_args)
    else:
        data = ModelData(cls=type(model),
                         init_args={'data': model})
    try:
        SAVED_DATA[model] = data
    except TypeError:
        pass
    # save variables only in the root model
    if SAVER_COUNTER == 1 and include_variables:
        print('saving variables in {}'.format(data))
        variables = twodlearn.core.common.get_variables(model)
        data.variables = [get_save_data(var) for var in variables]
    # exit node
    SAVER_COUNTER -= 1
    assert SAVER_COUNTER >= 0, 'SAVER_COUNTER resulted in negative value'
    return data


class OutputModelData(object):
    def __init__(self, model, feval, inputs=None):
        self.model = model
        self.feval = feval
        self.inputs = (inputs if inputs is not None
                       else dict())

    @twodlearn.core.common.LazzyProperty
    def _loaded(self):
        inputs = {key: load_data(value)
                  for key, value in self.inputs.items()}
        return getattr(load_data(self.model), self.feval)(**inputs)

    def load(self):
        return self._loaded


@register_saver(twodlearn.core.common.OutputModel, OutputModelData)
def _get_save_data_output_model(self):
    inputs = {arg: get_save_data(getattr(self, arg))
              for arg in self._inputs}
    return OutputModelData(model=get_save_data(self.model),
                           feval=self._feval.__name__,
                           inputs=inputs)
