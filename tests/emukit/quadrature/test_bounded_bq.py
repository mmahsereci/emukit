import GPy
import pytest
import numpy as np
from numpy.testing import assert_allclose
from math import isclose
from pytest_lazyfixture import lazy_fixture

from emukit.quadrature.methods.bounded_bq_model import BoundedBQModel
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFIsoGaussMeasure
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure
from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, BaseGaussianProcessGPy

REL_TOL = 1e-5
ABS_TOL = 1e-4


@pytest.fixture
def base_gp_wrong_kernel():
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    D = X.shape[1]
    integral_bounds = [(-1, 2), (-3, 3)]

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=D))
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), integral_bounds=integral_bounds)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return model


@pytest.fixture
def base_gp():
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    D = X.shape[1]
    measure = IsotropicGaussianMeasure(mean=np.array([0.1, 1.8]), variance=0.8)

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=D))
    qrbf = QuadratureRBFIsoGaussMeasure(RBFGPy(gpy_model.kern), measure=measure)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return model


@pytest.fixture
def bounded_bq_lower(base_gp):
    bound = np.min(base_gp.Y) - 0.5  # make sure bound is 0.5 below the Y values
    bounded_bq = BoundedBQModel(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, bound=bound, is_lower_bounded=True)
    return bounded_bq, bound


@pytest.fixture
def bounded_bq_upper(base_gp):
    bound = np.max(base_gp.Y) + 0.5  # make sure bound is 0.5 above the Y values
    bounded_bq = BoundedBQModel(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, bound=bound, is_lower_bounded=False)
    return bounded_bq, bound


models_test_list = [lazy_fixture("bounded_bq_lower"), lazy_fixture("bounded_bq_upper")]


@pytest.mark.parametrize('bounded_bq', models_test_list)
def test_bounded_bq_shapes(bounded_bq):
    model, _ = bounded_bq

    # integrate
    res = model.integrate()
    assert len(res) == 2
    assert isinstance(res[0], float)
    # None is returned temporarily until the variance is implemented.
    assert res[1] is None

    # transformations
    Y = np.array([[1], [2], [3]])
    assert model.transform(Y).shape == Y.shape
    assert model.inverse_transform(Y).shape == Y.shape

    # predictions base
    x = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 4]])

    res = model.predict_base(x)
    assert len(res) == 4
    for i in range(4):
        assert res[i].shape == (x.shape[0], 1)

    # predictions base full covariance
    res = model.predict_base_with_full_covariance(x)
    assert len(res) == 4
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], x.shape[0])
    assert res[2].shape == (x.shape[0], 1)
    assert res[3].shape == (x.shape[0], x.shape[0])

    # predictions
    res = model.predict(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], 1)

    # predictions full covariance
    res = model.predict_with_full_covariance(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], x.shape[0])

    # predict gradients
    res = model.get_prediction_gradients(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], x.shape[1])
    assert res[1].shape == (x.shape[0], x.shape[1])


@pytest.mark.parametrize('bounded_bq', models_test_list)
def test_bounded_bq_correct_bound(bounded_bq):
    model, bound = bounded_bq
    assert model.bound == bound


@pytest.mark.parametrize('bounded_bq', models_test_list)
def test_bounded_bq_transformations(bounded_bq):
    model, _ = bounded_bq

    # check if warping and inverse warping are correct yield identity.
    Y = np.array([[1], [2], [3]])
    assert_allclose(model.inverse_transform(model.transform(Y)), Y)
    assert_allclose(model.transform(model.inverse_transform(Y)), Y)

    # check if warping between base GP and model is consistent.
    Y2 = model.Y
    Y1 = model.base_gp.Y
    assert_allclose(model.transform(Y1), Y2)
    assert_allclose(model.inverse_transform(Y2), Y1)


def test_bounded_bq_raises_exception(base_gp_wrong_kernel):
    # wrong kernel embedding
    with pytest.raises(ValueError):
        BoundedBQModel(base_gp=base_gp_wrong_kernel, X=base_gp_wrong_kernel.X, Y=base_gp_wrong_kernel.Y,
                       bound=np.min(base_gp_wrong_kernel.Y) - 0.5, is_lower_bounded=True)

#def test_vanilla_bq_integrate(vanilla_bq):
#    # to check the integral, we check if it lies in some confidence interval.
#    # these intervals were computed as follows: the mean vanilla_bq.predict (first argument) was integrated by
#    # simple random sampling with 1e6 samples, and the variance (second argument) with 5*1e3 samples. This was done 100
#    # times. The intervals show mean\pm 3 std of the 100 integrals obtained by sampling. There might be a very small
#    # chance the true integrals lies outside the specified intervals.
#    interval_mean = [10.020723475428762, 10.09043533562786]
#    interval_var = [41.97715934990283, 46.23549367612568]

#    integral_value, integral_variance = vanilla_bq.integrate()
#    assert interval_mean[0] < integral_value < interval_mean[1]
#    assert interval_var[0] < integral_variance < interval_var[1]


@pytest.mark.parametrize('bounded_bq', models_test_list)
def test_bounded_bq_gradients(bounded_bq):
    model, _ = bounded_bq
    D = model.X.shape[1]
    N = 4
    x = np.reshape(np.random.randn(D * N), [N, D])

    # mean
    mean_func = lambda z: model.predict(z)[0]
    mean_grad_func = lambda z: model.get_prediction_gradients(z)[0]
    _check_grad(mean_func, mean_grad_func, x)

    # var
    var_func = lambda z: model.predict(z)[1]
    var_grad_func = lambda z: model.get_prediction_gradients(z)[1]
    _check_grad(var_func, var_grad_func, x)


def _compute_numerical_gradient(func, grad_func, x, eps=1e-6):
    f = func(x)
    grad = grad_func(x)

    grad_num = np.zeros(grad.shape)
    for d in range(x.shape[1]):
        x_tmp = x.copy()
        x_tmp[:, d] = x_tmp[:, d] + eps
        f_tmp = func(x_tmp)
        grad_num_d = (f_tmp - f) / eps
        grad_num[:, d] = grad_num_d[:, 0]
    return grad, grad_num


def _check_grad(func, grad_func, x):
    grad, grad_num = _compute_numerical_gradient(func, grad_func, x)
    isclose_all = 1 - np.array([isclose(grad[i, j], grad_num[i, j], rel_tol=REL_TOL, abs_tol=ABS_TOL)
                                for i in range(grad.shape[0]) for j in range(grad.shape[1])])
    assert isclose_all.sum() == 0