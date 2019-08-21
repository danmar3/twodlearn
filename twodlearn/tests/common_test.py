# import logging
# logging.getLogger('tensorflow').disabled = True
import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet.bayesnet
import twodlearn.bayesnet.gaussian_process
import twodlearn.templates.bayesnet


class CommonTest(unittest.TestCase):
    def test_rank(self):
        assert tdl.core.tensor_rank(np.array([[1.0]])) == 0,\
            'Error on computing tensor_rank'
        assert tdl.core.tensor_rank(np.array([1.0])) == 0,\
            'Error on computing tensor_rank'
        assert tdl.core.tensor_rank(np.array([[1, 2, 3],
                                              [4, 5, 6]])) == 2,\
            'Error on computing tensor_rank'
        assert tdl.core.tensor_rank(np.array([[1, 2, 3]])) == 1,\
            'Error on computing tensor_rank'

    def test_eager(self):
        class TestProgram(object):
            @tdl.core.EagerMethod
            def eager_fn(self, alpha):
                self._alpha = alpha

            @eager_fn.eval
            def eager_fn(self, x):
                return self._alpha * x
        scale = [2.0, 10.5]
        x = [5.0, 8.0]
        test = [TestProgram(), TestProgram()]
        test[0].eager_fn = [scale[0]]
        test[1].eager_fn = [scale[1]]
        for i in range(len(test)):
            assert test[i]._alpha == scale[i],\
                'eager function is not setting attributes correctly'
            assert x[i]*scale[i] == test[i].eager_fn(x[i]),\
                'incorrect evaluation of eager function'

    def test_simple_parameter(self):
        class TestClass(object):
            @tdl.core.SimpleParameter
            def alpha(self, value):
                return value

        alpha = [2.0, 10.5]
        test = [TestClass(), TestClass()]
        for i in range(len(test)):
            test[i].alpha = alpha[i]
        for i in range(len(test)):
            assert test[i].alpha == alpha[i],\
                'eager function is not setting attributes correctly'

    def test_tdl_program(self):
        class TestProgram(tdl.core.TdlProgram):
            @tdl.core.EagerMethod
            def eager_fn(self, alpha):
                self._alpha = alpha

            @eager_fn.eval
            def eager_fn(self, x):
                return self._alpha * x
        alpha = [2.0, 10.5]
        x = [5.0, 8.0]
        test = [TestProgram(eager_fn=arg) for arg in alpha]
        for i in range(len(test)):
            assert test[i]._alpha == alpha[i],\
                'eager function is not setting attributes correctly'
            assert x[i]*alpha[i] == test[i].eager_fn(x[i]),\
                'incorrect evaluation of eager function'

        alpha = [2.0, 10.5]
        x = [5.0, 8.0]
        test = [TestProgram(eager_fn=[arg]) for arg in alpha]
        for i in range(len(test)):
            assert test[i]._alpha == alpha[i],\
                'eager function is not setting attributes correctly'
            assert x[i]*alpha[i] == test[i].eager_fn(x[i]),\
                'incorrect evaluation of eager function'

    def test_find_attrs(self):
        model = tdl.bayesnet.bayesnet.BernoulliBayesianMlp(10, 10, [])
        assert set(['layers']) == set(model.__tdl__._submodels),\
            'Incorrect tdl attributes have been found'
        assert set([]) == set(model.__tdl__._parameters),\
            'Incorrect tdl attributes have been found'

        model = tdl.bayesnet.gaussian_process.GaussianProcess(
            xm=np.random.normal(size=[10, 5]).astype(np.float32),
            ym=np.random.normal(size=[1, 10]).astype(np.float32))
        assert (set(['kernel']) ==
                set(model.__tdl__._submodels)),\
            'Incorrect tdl attributes have been found'
        assert (set(['l_scale', 'f_scale', 'y_scale']) ==
                set(model.kernel.__tdl__._parameters)),\
            'Incorrect tdl attributes have been found'

    def test_find_attrs2(self):
        class TestInit0(tdl.core.TdlModel):
            @tdl.core.InputArgument
            def one(self, value):
                return value

        class TestInit1(TestInit0):
            @tdl.core.InputArgument
            def two(self, value):
                return value

        test = TestInit1(one=1, two=2)
        assert test.one == 1 and test.two == 2

    def test_tdlobject(self):
        class TestProgram(tdl.core.TdlObject):
            @tdl.core.EncapsulatedMethod
            def local_counter(self, local, value):
                local.alpha = 0

            @local_counter.eval
            def local_counter(self, local):
                local.alpha += 1
                return local.alpha

        obj = [TestProgram() for i in range(3)]
        for i in range(len(obj)):
            for j in range(i):
                obj[i].local_counter()

        for i in range(len(obj)):
            assert (obj[i].local_counter() == i+1),\
                'error while evaluating local counter'

    def test_tdlobject_local(self):
        class TestProgram(tdl.core.TdlObject):
            @tdl.core.EncapsulatedMethod
            def local_counter(self, local, value):
                local.alpha = 0

            @local_counter.eval
            def local_counter(self, local):
                local.alpha += 1
                return local.alpha

        obj = [TestProgram() for i in range(3)]
        for i in range(len(obj)):
            obj[i].local_counter.local.alpha = i

        for i in range(len(obj)):
            assert (obj[i].local_counter() == i+1),\
                'error while evaluating local counter'

    def test_optional_property(self):
        class TestObject(tdl.core.TdlObject):
            @tdl.core.OptionalProperty
            def regularizer(self, stddev):
                return stddev**2.0

        main_list = [TestObject() for i in range(5)]
        for i, main in enumerate(main_list):
            assert main.regularizer.value is None,\
                'optional property error'
            assert main.regularizer.is_set is False,\
                'optional property error'
            value = main.regularizer.init(i)
            assert value == i**2.0, \
                'optional property error'
            assert main.regularizer.value == i**2.0,\
                'optional property error'
            main.regularizer = i
            assert main.regularizer.value == i,\
                'optional property error'

    def test_submodelinit(self):
        class TestObject(tdl.core.TdlModel):
            @tdl.core.SubmodelInit
            def submodel(self, x, y):
                return tdl.core.SimpleNamespace(x=x, y=y)
        x = [1, 2, 3, 4]
        y = ['a', 'b', 'c', 'd']
        obj = [TestObject() for i in range(len(x))]
        for xi, yi, obj_i in zip(x, y, obj):
            obj_i.submodel.init(x=xi, y=yi)
        for xi, yi, obj_i in zip(x, y, obj):
            assert (obj_i.submodel.x == xi and obj_i.submodel.y == yi),\
                'SubmodelInit test failed'

    def test_lazzy_submodelinit(self):
        class TestObject(tdl.core.TdlModel):
            @tdl.core.SubmodelInit(lazzy=True)
            def submodel1(self, x, y):
                return tdl.core.SimpleNamespace(x=x, y=y)

            @tdl.core.SubmodelInit(lazzy=True)
            def submodel2(self, x):
                tdl.core.assert_initialized(self, 'submodel2', ['submodel1'])
                return self.submodel1.x*x

        x = [1, 2, 3, 4]
        y = ['a', 'b', 'c', 'd']
        z = [1.546, 354.52, 564.4, 54.3]
        obj = [TestObject() for i in range(len(x))]
        for xi, yi, zi, obj_i in zip(x, y, z, obj):
            obj_i.submodel1.init(x=xi, y=yi)
            obj_i.submodel2.init(x=zi)
        for xi, yi, zi, obj_i in zip(x, y, z, obj):
            assert (obj_i.submodel1.x == xi and obj_i.submodel1.y == yi),\
                'SubmodelInit test failed'
            assert obj_i.submodel2 == xi*zi, 'SubmodelInit test failed'

        # test init
        obj = [TestObject(submodel1={'x': x[i], 'y': y[i]})
               for i in range(len(x))]
        for obj_i in obj:
            assert not tdl.core.is_property_initialized(obj_i, 'submodel1')
            assert not tdl.core.is_property_initialized(obj_i, 'submodel2')

        for xi, yi, zi, obj_i in zip(x, y, z, obj):
            obj_i.submodel2.init(x=zi)
        for xi, yi, zi, obj_i in zip(x, y, z, obj):
            assert (obj_i.submodel1.x == xi and obj_i.submodel1.y == yi),\
                'SubmodelInit test failed'
            assert obj_i.submodel2 == xi*zi, 'SubmodelInit test failed'

    def test_inputmodelinit(self):
        class TestObject0(tdl.core.TdlObject):
            @tdl.core.InputModelInit
            def submodel(self, x, y):
                return tdl.core.SimpleNamespace(x=x, y=y)

        class TestObject1(tdl.core.TdlObject):
            submodel = tdl.core.InputModelInit(inference_input=True)

            @submodel.initializer
            def submodel(self, x, y):
                return tdl.core.SimpleNamespace(x=x, y=y)

        class TestObject2(tdl.core.TdlObject):
            @tdl.core.InputModelInit(inference_input=True)
            def submodel(self, x, y):
                return tdl.core.SimpleNamespace(x=x, y=y)

        class TestObject3(TestObject2):
            pass

        def test_initialization(ObjClass):
            x = [1, 2, 3, 4]
            y = ['a', 'b', 'c', 'd']
            obj = [ObjClass() for i in range(len(x))]
            for xi, yi, obj_i in zip(x, y, obj):
                obj_i.submodel.init(x=xi, y=yi)
            for xi, yi, obj_i in zip(x, y, obj):
                assert (obj_i.submodel.x == xi and obj_i.submodel.y == yi),\
                    'InputModelInit test failed'

        [test_initialization(ObjClass)
         for ObjClass in (TestObject0, TestObject1, TestObject2, TestObject3)]

    def test_inputmodelinit2(self):
        class TestObject0(tdl.core.TdlModel):
            @tdl.core.InputModelInit
            def submodel(self, x=None, y=None):
                return tf.placeholder(tf.float32)

        class TestObject1(tdl.core.TdlModel):
            submodel = tdl.core.InputModelInit(inference_input=True)

            @submodel.initializer
            def submodel(self, x=None, y=None):
                return tf.placeholder(tf.float32)

        class TestObject2(tdl.core.TdlModel):
            @tdl.core.InputModelInit(inference_input=True)
            def submodel(self, x=None, y=None):
                return tf.placeholder(tf.float32)

        class TestObject3(TestObject2):
            pass

        objects = [ObjClass(submodel=tdl.AutoInit())
                   for ObjClass in (TestObject0, TestObject1, TestObject2,
                                    TestObject3)]

        assert not tdl.core.get_placeholders(objects[0])
        for obj in (objects[1], objects[2], objects[3]):
            assert isinstance(tdl.core.get_placeholder(obj), tf.Tensor)

    def test_is_property_set(self):
        model = twodlearn.templates.bayesnet.GpEstimator()
        assert not tdl.core.is_property_set(model, 'model')
        assert not tdl.core.is_property_set(model, 'optimizer')

    def test_disableautoinit(self):
        class TestObject(tdl.core.TdlModel):
            _input_args = ['four', 'two', 'one']

            @tdl.core.InputArgument
            def one(self, value):
                return value

            @tdl.core.InputArgument
            def two(self, value):
                return (self.one + value if None not in (self.one, value)
                        else None)

            @tdl.core.InputArgument
            def three(self, value):
                # should raise exeption
                return self.submodel

            @tdl.core.InputArgument
            def four(self, value):
                tdl.core.assert_initialized(self, 'four', ['one', 'two'])
                return value

            @tdl.core.Submodel
            def submodel(self, value):
                return (self.one, self.two, value)

        test_1 = TestObject(one=1, two=1)
        assert test_1.two == 2, 'error with initialization'

        # ------ assert exeption is raised -----
        with self.assertRaises(tdl.core.exceptions.InitPreconditionsFailed):
            test_2 = TestObject()
            with tdl.core.DisableAutoinit(test_2):
                test_2.two = 1

        # ------ test initialization of submodels -----
        test_3 = TestObject(submodel=3)
        assert test_3.submodel == (None, None, 3),\
            'error initializing submodel'

        # ------ assert exeption is raised -----
        with self.assertRaises(tdl.core.exceptions.InitPreconditionsFailed):
            test_4 = TestObject(three=3)

        test_5 = TestObject(four=4, one=1)
        assert (test_5.one, test_5.two, test_5.four) == (1, None, 4),\
            'error initializing model'

    def test_autoinit(self):
        class TestObject(tdl.core.TdlModel):
            _input_args = ['four', 'two', 'one']

            @tdl.core.InputArgument
            def one(self, value):
                return value

            @tdl.core.InputArgument
            def two(self, value):
                return (self.one + value if None not in (self.one, value)
                        else None)

            @tdl.core.InputArgument
            def three(self, value):
                # should raise exeption
                return self.submodel

            @tdl.core.InputArgument
            def four(self, value):
                tdl.core.assert_initialized(self, 'four', ['two'])
                return value

            @tdl.core.Submodel
            def submodel(self, value):
                return (self.one, self.two, value)

            @tdl.core.SubmodelInit
            def submodel1(self, value1, value2):
                return value1

            @tdl.core.SubmodelInit
            def submodel2(self, value1, value2):
                tdl.core.assert_initialized(self, 'submodel2', ['submodel1'])
                return value1

        test = TestObject()
        assert test.four is None, 'autoinit failed'
        with tdl.core.DisableAutoinit(test):
            assert (test.one, test.two) == (None, None),\
                'autoinit failed'

        with self.assertRaises(tdl.core.exceptions.InitPreconditionsFailed):
            test.submodel2.init(1, 2)

        with self.assertRaises(tdl.core.exceptions.PropertyRedefinition):
            test.one = 1

    def test_autoinit2(self):
        with self.assertRaises(tdl.core.exceptions.NonePropertyAvailable):
            mvn1 = tdl.bayesnet.distributions.MVNScaledIdentity(scale=0.5)

    def test_nottrainable(self):
        mvn1 = tdl.bayesnet.distributions.MVNScaledIdentity(
            shape=[10], scale=0.5)
        with tdl.core.NotTrainable():
            mvn2 = tdl.bayesnet.distributions.MVNScaledIdentity(
                shape=[10], scale=0.5)
            tdl.core.init_attrs(mvn2)
        with tdl.core.NotTrainable():
            mvn3 = tdl.bayesnet.distributions.MVNScaledIdentity(
                shape=[10], scale=0.5, loc=tdl.AutoInit())
        assert tdl.core.is_trainable(mvn1.loc)
        assert tdl.core.is_trainable(mvn1.covariance.raw)
        assert not tdl.core.is_trainable(mvn2.loc)
        assert not tdl.core.is_trainable(mvn2.covariance.raw)
        assert not tdl.core.is_trainable(mvn3.loc)
        assert not tdl.core.is_trainable(mvn3.covariance.raw)

    def test_nottrainable2(self):
        mvn1 = tdl.bayesnet.distributions.MVNScaledIdentity(
            shape=[10], scale=(0.5, tdl.AutoTensor()), loc=tdl.AutoTensor())
        assert not tdl.core.is_trainable(mvn1.loc)
        assert not tdl.core.is_trainable(mvn1.covariance.raw)
        assert np.trace(mvn1.covariance.value.eval()) == (0.5*0.5)*10
        mvn2 = tdl.bayesnet.distributions.MVN(
            shape=[10], scale=(0.5, tdl.AutoTensor()), loc=tdl.AutoTensor())
        assert not tdl.core.is_trainable(mvn2.loc)
        assert not tdl.core.is_trainable(mvn2.covariance.raw)

    def test_autotype(self):
        class TestModel(tdl.core.TdlModel):
            @tdl.core.InputArgument
            def one(self, value, AutoType=None):
                if AutoType is None:
                    AutoType = int
                if value is None:
                    value = 1
                return AutoType(value)

            @tdl.core.SimpleParameter
            def two(self, value, AutoType=None):
                if AutoType is None:
                    AutoType = int
                if value is None:
                    value = 2
                return AutoType(value)

        m1 = TestModel()
        assert m1.one == 1 and type(m1.one) is int
        m2 = TestModel(one=2)
        assert m2.one == 2 and type(m2.one) is int
        m3 = TestModel(one=(3, float))
        assert m3.one == 3 and type(m3.one) is float

    def test_shortcuts(self):
        class TestModel1(tdl.core.TdlModel):
            @tdl.core.InputArgument
            def one(self, value):
                return (value if value is not None
                        else 1)

            @tdl.core.InputArgument
            def two(self, value):
                return (value if value is not None
                        else 2)

        @tdl.core.PropertyShortcuts({'model': ['one', 'two']})
        class TestModel2(tdl.core.TdlModel):
            @tdl.core.InputArgument
            def model(self, value):
                return value

        model1 = TestModel1()
        model2 = TestModel2(model=model1)
        assert model2.one == 1
        assert model2.two == 2

    def test_assert_any_available(self):
        class TestModel1(tdl.core.TdlModel):
            @tdl.core.InputArgument
            def one(self, value):
                return value

            @tdl.core.InputArgument
            def two(self, value):
                return value

            @tdl.core.LazzyProperty
            def three(self):
                tdl.core.assert_any_available(self, reqs=['one', 'two'])
                if tdl.core.is_property_set(self, 'one'):
                    return self.one
                elif tdl.core.is_property_set(self, 'two'):
                    return self.two
                else:
                    raise AssertionError('assert_any_available failed')

            @tdl.core.InputArgument
            def four(self, value):
                tdl.core.assert_any_available(self, reqs=['one', 'two'])
                if tdl.core.is_property_set(self, 'one'):
                    return value + self.one
                elif tdl.core.is_property_set(self, 'two'):
                    return value + self.two
                else:
                    raise AssertionError('assert_any_available failed')

        model1 = TestModel1(one=1)
        assert model1.three == 1
        model2 = TestModel1(two=2)
        assert model2.three == 2
        model3 = TestModel1(one=1, two=2)
        assert model3.three == 1
        model4 = TestModel1()
        with self.assertRaises(tdl.core.exceptions.NonePropertyAvailable):
            test = model4.three
        with self.assertRaises(tdl.core.exceptions.NonePropertyAvailable):
            model5 = TestModel1(four=4)

    def test_assert_initialized_if_available(self):
        class TestModel1(tdl.core.TdlModel):
            _input_args = ['three', 'two', 'one']

            @tdl.core.InputArgument
            def one(self, value):
                return value

            @tdl.core.InputArgument
            def two(self, value):
                return value

            @tdl.core.InputArgument
            def three(self, value):
                tdl.core.assert_initialized_if_available(
                    self, 'three', ['one', 'two'])
                if tdl.core.is_property_set(self, 'one'):
                    return value + self.one
                elif tdl.core.is_property_set(self, 'two'):
                    return value + self.two
                else:
                    return value

        model1 = TestModel1(one=1, three=3)
        assert model1.three == 4
        assert not tdl.core.is_property_set(model1, 'two')
        model2 = TestModel1(three=3)
        assert model2.three == 3
        assert not tdl.core.is_property_set(model2, 'one')
        assert not tdl.core.is_property_set(model2, 'two')
        model3 = TestModel1(two=2, three=3)
        assert model3.three == 5
        assert not tdl.core.is_property_set(model3, 'one')

    def test_hasattr(self):
        class TestClass1(tdl.core.TdlModelCallable):
            @tdl.core.InputArgument
            def one(self, value):
                return value

        class TestClass2(tdl.core.TdlModelCallable):
            @tdl.core.InputArgument
            def one(self, value):
                return value

            @tdl.core.InputArgument
            def two(self, value):
                return value

        test1 = TestClass1()
        assert tdl.core.hasattr(test1, 'one')
        assert not tdl.core.is_property_initialized(test1, 'one')
        assert not tdl.core.hasattr(test1, 'two')
        test2 = TestClass2()
        test2.one = 1
        assert tdl.core.hasattr(test2, 'one')
        assert tdl.core.is_property_initialized(test2, 'one')
        assert tdl.core.hasattr(test2, 'two')
        assert not tdl.core.is_property_initialized(test2, 'two')

    def test_conditional_initialization(self):
        @tdl.core.create_init_docstring
        class TestClass(tdl.stacked.StackedLayers):
            @tdl.core.InputArgument
            def input_shape(self, value):
                if value is None:
                    tdl.core.assert_initialized_if_available(
                        self, 'input_shape', ['embedding_size'])
                    if tdl.core.is_property_initialized(
                            self, 'embedding_size'):
                        value = tf.TensorShape([None, self.embedding_size])
                    else:
                        raise tdl.core.exceptions.ArgumentNotProvided(self)
                return tf.TensorShape(value)

            @tdl.core.InputArgument
            def embedding_size(self, value):
                if value is None:
                    tdl.core.assert_initialized(
                        self, 'embedding_size', ['input_shape'])
                    value = self.input_shape[-1].value
                return value

        test1 = TestClass(embedding_size=5)
        assert test1.input_shape.as_list() == [None, 5]
        test2 = TestClass(input_shape=[None, 5])
        assert test2.embedding_size == 5
        test3 = TestClass(input_shape=[None, 5])
        assert tdl.core.any_initialized(
            test3, ['input_shape', 'embedding_size'])

    def test_check_tdl_args(self):
        class TestClass(tdl.core.TdlModel):
            def _tdl_check_kwargs(self, kwargs):
                if 'input1' in kwargs and 'input2' in kwargs:
                    raise ValueError('input1 and input2 cannot be specified'
                                     'at the same time')

            @tdl.core.InputArgument
            def input1(self, value):
                return value

            @tdl.core.InputArgument
            def input2(self, value):
                return value

        test1 = TestClass(input1=1)
        test2 = TestClass(input2=1)
        with self.assertRaises(ValueError):
            test3 = TestClass(input1=1, input2=2)

    def test_descriptor_shortcut(self):
        class GMM(tdl.core.layers.Layer):
            n_dims = tdl.core.InputArgument.required(
                'n_dims', doc='dimensions of the GMM model')
            n_components = tdl.core.InputArgument.required(
                 'n_components', doc='number of mixture components')
            trainable = tdl.core.InputArgument.optional(
                 'trainable', doc='is trainable')

        test1 = GMM(n_dims=1, n_components=2)
        test2 = GMM(n_dims=10, n_components=20)
        assert (test1.n_dims == 1
                and test1.n_components == 2
                and test1.trainable is None)
        assert (test2.n_dims == 10 and test2.n_components == 20)
        with self.assertRaises(ValueError):
            test3 = GMM(n_dims=10)
            test3.n_components


if __name__ == "__main__":
    unittest.main()
