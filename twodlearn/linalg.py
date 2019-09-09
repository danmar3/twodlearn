import tensorflow as tf
import twodlearn as tdl


def solvevec(M_cholesky, x):
    """compute inv(M) @ x

    Args:
        M_cholesky : cholesky decomposition of matrix M.
        x (tf.Tensor): (batched) vector
    Returns:
        tf.Tensor: solution of inv(M) @ x
    """
    x = tf.convert_to_tensor(x)
    return M_cholesky.solvevec(
        M_cholesky.solvevec(x),
        adjoint=True)


def solvemat(M_cholesky, A):
    """Compute inv(M) @ A

    Args:
        M_cholesky : cholesky decomposition of matrix M
        A (tf.Tensor): (batched) Matrix.
    Returns:
        tf.Tensor: solution of inv(M) @ A
    """
    A = tf.convert_to_tensor(A)
    return M_cholesky.solve(
        M_cholesky.solve(A),
        adjoint=True)


def is_diagonal_linop(M):
    return isinstance(M, (tf.linalg.LinearOperatorDiag,
                          tf.linalg.LinearOperatorIdentity,
                          tf.linalg.LinearOperatorScaledIdentity))


def diagonal_M_times_Mt(M):
    """Compute M @ M^t for diagonal linop M.

    Args:
        M : (batched) diagonal linear operator
    Returns:
        tf.Tensor: solution of M @ M^t
    """
    assert is_diagonal_linop(M)
    if isinstance(M, tf.linalg.LinearOperatorScaledIdentity):
        num_rows = (M.domain_dimension.value
                    if M.domain_dimension.value is not None
                    else M.domain_dimension_tensor())
        return tf.linalg.LinearOperatorScaledIdentity(
            num_rows=num_rows,
            multiplier=tf.square(M.multiplier)
        )
    elif isinstance(M, tf.linalg.LinearOperatorDiag):
        return tf.linalg.LinearOperatorDiag(diag=tf.square(M.diag))
    elif isinstance(M, tf.linalg.LinearOperatorIdentity):
        return M


def M_times_Mt(M):
    """Compute M @ M^t

    Args:
        M : (batched) matrix M
    Returns:
        tf.Tensor: solution of M @ M^t
    """
    if isinstance(M, tf.Tensor):
        linop = tf.linalg.LinearOperatorFullMatrix(M)
        return linop.matmul(M, adjoint_arg=True)
    elif isinstance(M, (tf.linalg.LinearOperatorFullMatrix,
                        tf.linalg.LinearOperatorLowerTriangular)):
        return M.matmul(M.to_dense(), adjoint_arg=True)
    elif is_diagonal_linop(M):
        return diagonal_M_times_Mt(M)
    else:
        raise TypeError("cannot compute M_times_Mt, invalid type")


def Mt_times_M(M):
    """Compute M^t @ M

    Args:
        M : (batched) matrix M
    Returns:
        tf.Tensor: solution of M^t @ M
    """
    if isinstance(M, tf.Tensor):
        linop = tf.linalg.LinearOperatorFullMatrix(M)
        return linop.matmul(M, adjoint=True)
    elif isinstance(M, (tf.linalg.LinearOperatorFullMatrix,
                        tf.linalg.LinearOperatorLowerTriangular)):
        return M.matmul(M.to_dense(), adjoint=True)
    elif is_diagonal_linop(M):
        return diagonal_M_times_Mt(M)
    else:
        raise TypeError("cannot compute M_times_Mt, invalid type")


@tdl.core.create_init_docstring
class PDMatrix(tdl.core.TdlModel):
    ''' Positive definite matrix '''
    @tdl.core.InputArgument
    def batch_shape(self, value):
        if value is None and tdl.core.is_property_set(self, 'shape'):
            return self.shape[:-2]
        elif tdl.core.is_property_set(self, 'shape'):
            raise ValueError('batch_shape and shape cannot be '
                             'specified at the same time')
        elif value is None:
            value = ()
        return tf.TensorShape(value)

    @tdl.core.InputArgument
    def domain_dimension(self, value):
        if value is None and tdl.core.is_property_set(self, 'shape'):
            return self.shape[-1]
        elif tdl.core.is_property_set(self, 'shape'):
            raise ValueError('domain dimension and shape cannot be '
                             'specified at the same time')
        return tf.Dimension(value)

    @tdl.core.InputArgument
    def shape(self, value):
        ''' shape of the distribution.
        The shape assumes the last two dimentions corresponds to the shape of a
        square matrix
        '''
        if value is None:
            if tdl.core.is_property_set(self, 'raw'):
                return tf.convert_to_tensor(self.raw).shape
            tdl.core.assert_initialized(self, 'shape',
                                        ['batch_shape', 'domain_dimension'])
            return self.batch_shape.concatenate(
                [self.domain_dimension]*2)
        elif isinstance(value, int):
            value = [value, value]
        if isinstance(value, list):
            value = tf.TensorShape(value)
        return value

    @tdl.core.InputArgument
    def tolerance(self, value):
        return value

    @tdl.core.SimpleParameter
    def raw(self, value, AutoType=None):
        if AutoType is None:
            AutoType = tdl.core.variable
        if value is None:
            shape = self.shape.as_list()
            if shape[-1] is None:
                raise ValueError('Unable to initialize {}, shape not specified'
                                 ''.format(self))
            eye = tf.eye(shape[-1], batch_shape=shape[:-2])
            value = AutoType(eye)
        elif isinstance(value, (int, float)):
            shape = self.shape.as_list()
            if shape[-1] is None:
                raise ValueError('Unable to initialize {}, shape not specified'
                                 ''.format(self))
            eye = tf.eye(shape[-1], batch_shape=shape[:-2])
            value = AutoType(value*eye)
        return value

    @tdl.core.LazzyProperty
    def linop(self):
        scale = self.raw
        if self.tolerance is not None:
            scale = tdl.core.array.add_diagonal_shift(scale, self.tolerance)
        scale_t = tdl.core.array.transpose_rightmost(scale)
        value = tf.matmul(scale, scale_t)
        return tf.linalg.LinearOperatorFullMatrix(value)

    @tdl.core.LazzyProperty
    def cholesky(self):
        return Cholesky(self.linop)

    @tdl.core.OutputValue
    def value(self, value):
        tdl.core.assert_initialized(self, 'value', ['linop'])
        if value is None:
            value = self.linop.to_dense()
        return value


@tdl.core.create_init_docstring
class PDMatrixDiag(PDMatrix):
    @tdl.core.SimpleParameter
    def raw(self, value, AutoType=None):
        ''' square root of the diagonal part for the matrices '''
        if AutoType is None:
            AutoType = tdl.core.variable
        if value is None:
            shape = self.batch_shape.concatenate(self.domain_dimension)
            value = AutoType(tf.ones(shape=shape))
        elif isinstance(value, (int, float)):
            shape = self.batch_shape.concatenate(self.domain_dimension)
            value = AutoType(value*tf.ones(shape=shape))
        return value

    @tdl.core.LazzyProperty
    def linop(self):
        scale = self.raw
        if self.tolerance is not None:
            scale = scale + self.tolerance
        return tf.linalg.LinearOperatorDiag(tf.square(scale))


@tdl.core.create_init_docstring
class PDScaledIdentity(PDMatrixDiag):
    @tdl.core.InputArgument
    def domain_dimension(self, value):
        if value is None:
            tdl.core.assert_initialized(self, 'domain_dimension', ['shape'])
            return self.shape[-1]
        elif tdl.core.is_property_set(self, 'shape'):
            raise ValueError('domain dimension and shape cannot be '
                             'specified at the same time')
        elif isinstance(value, tf.Tensor):
            tdl.core.assert_initialized(self, 'domain_dimension', ['raw'])
            self.linop.init(domain_dimension=value)
            return tf.Dimension(None)
        return tf.Dimension(value)

    @tdl.core.InputParameter
    def raw(self, value, AutoType=None):
        ''' multiplier for the identities '''
        if AutoType is None:
            AutoType = tdl.core.variable
        if value is None:
            value = AutoType(tf.ones(shape=self.batch_shape))
        elif isinstance(value, (int, float)):
            value = AutoType(tf.fill(dims=self.batch_shape, value=value))
        return value

    @tdl.core.SubmodelInit
    def linop(self, domain_dimension=None):
        if domain_dimension is None:
            tdl.core.assert_initialized(self, 'linop', ['domain_dimension'])
            domain_dimension = self.domain_dimension
        scale = self.raw
        if self.tolerance is not None:
            scale = scale + self.tolerance
        return tf.linalg.LinearOperatorScaledIdentity(
            num_rows=domain_dimension,
            multiplier=tf.square(scale))


@tdl.core.create_init_docstring
class Cholesky(tdl.core.TdlModel):
    @tdl.core.InputParameter
    def input(self, value):
        return value

    @tdl.core.LazzyProperty
    def linop(self):
        input = (self.input.linop if isinstance(self.input, PDMatrix)
                 else self.input)
        if is_diagonal_linop(input):
            return tf.linalg.LinearOperatorDiag(
                tf.sqrt(input.diag_part()))
        elif isinstance(input, tf.linalg.LinearOperatorFullMatrix):
            self.value = tf.linalg.cholesky(input.to_dense())
        else:
            self.value = tf.linalg.cholesky(input)
        return tf.linalg.LinearOperatorLowerTriangular(self.value)

    @tdl.core.Submodel
    def value(self, value):
        if value is None:
            value = self.linop.to_dense()
        return value

    def log_abs_determinant(self):
        tdl.core.assert_initialized(self, 'log_abs_determinant', ['linop'])
        return self.linop.log_abs_determinant()

    def solve(self, rhs, adjoint=False, adjoint_arg=False, name='solve'):
        return self.linop.solve(rhs=rhs,
                                adjoint=adjoint,
                                adjoint_arg=adjoint_arg,
                                name=name)

    def solvevec(self, rhs, adjoint=False, name='solve'):
        return self.linop.solvevec(rhs=rhs, adjoint=adjoint, name=name)

    def matmul(self, x, adjoint=False, adjoint_arg=False, name='matmul'):
        return self.linop.matmul(x=x, adjoint=adjoint, adjoint_arg=adjoint_arg,
                                 name=name)

    def matvec(self, x, adjoint=False, name='matvec'):
        return self.linop.matvec(x=x, adjoint=adjoint, name=name)

    def diag_part(self, name="diag_part"):
        return self.linop.diag_part(name=name)

    def shape_tensor(self):
        ''' Shape determined at runtime '''
        return self.linop.shape_tensor()

    def __init__(self, input, name=None):
        super(Cholesky, self).__init__(input=input, name=name)


def is_diagonal(x):
    if isinstance(x, (Cholesky, PDMatrix)):
        return is_diagonal_linop(x.linop)
    elif isinstance(x, (tf.linalg.LinearOperator)):
        return is_diagonal_linop(x)
    else:
        raise TypeError('Unrecognized type {} for is_diagonal method'
                        ''.format(type(x)))


def is_scaled_identity(x):
    if not is_diagonal(x):
        return False
    if isinstance(x, tf.linalg.LinearOperator):
        linop = x
    elif isinstance(x, (Cholesky, PDMatrix)):
        linop = x.linop
    else:
        raise TypeError('Unrecognized type {} for is_diagonal method'
                        ''.format(type(x)))
    return isinstance(linop, (tf.linalg.LinearOperatorScaledIdentity))


@tdl.core.create_init_docstring
class DynamicScaledIdentity(tdl.core.TdlModel):
    '''scaled identity [batched] matrix whose size depends on the
    shape of the input'''
    @property
    def batch_shape(self):
        return tf.convert_to_tensor(self.multipliers).shape

    @tdl.core.InputArgument
    def tolerance(self, value):
        return value

    @tdl.core.InputParameter
    def multipliers(self, value):
        """tensor whose elements represent the multiplier factors for the
        identity matrices.
        """
        return value

    def __call__(self, inputs):
        '''returns a linear scaled identity operator.
        The number of rows in the identity matrices is equal to
        inputs.shape[-2]
        '''
        inputs = tf.convert_to_tensor(inputs)
        if inputs.shape.is_fully_defined():
            n_rows = inputs.shape[-2]
        else:
            n_rows = tf.shape(inputs)[-2]
        if self.tolerance is None:
            multipliers = self.multipliers
        else:
            multipliers = self.multipliers + self.tolerance
        return tf.linalg.LinearOperatorScaledIdentity(
            num_rows=n_rows, multiplier=multipliers
        )
