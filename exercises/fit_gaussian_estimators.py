from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


# Q2 Empirically showing sample mean is consistent
def _evaluate_expectation_error(samples, mu):
    ms = np.linspace(10, 1000, 100, dtype=int)
    expectation_error = []
    for m in ms:
        estimated = UnivariateGaussian().fit(samples[:m])
        expectation_error.append(abs(estimated.mu_ - mu))

    expectation_error_fig = go.Figure()
    expectation_error_fig.add_scatter(x=ms, y=expectation_error, mode='lines')
    expectation_error_fig.update_layout(dict(title="Expectation Error As Function Of Number Of Samples",
                                             xaxis_title="$m\\text{ - number of samples}$",
                                             yaxis_title="r$|\mu-\hat\mu|\\text{ - expectation error}$"))
    expectation_error_fig.show()


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    mu, sigma, num_samples = 10, 1, 1000
    samples = np.random.normal(mu, sigma, num_samples)
    univariate_gaussian = UnivariateGaussian()
    fitted = univariate_gaussian.fit(samples)
    print(f"({fitted.mu_}, {fitted.var_})")

    # Question 2 - Empirically showing sample mean is consistent

    _evaluate_expectation_error(samples, mu)

    # Question 3 - Plotting Empirical PDF of fitted model

    pdfs = univariate_gaussian.pdf(samples)
    pdfs_fig = go.Figure()
    pdfs_fig.add_scatter(x=samples, y=pdfs, mode="markers", marker=dict(color="red", size=1.5))
    pdfs_fig.update_layout(dict(title="Empirical PDF Function Under The Fitted Model",
                                xaxis_title="$x\\text{ - sample values}$",
                                yaxis_title="$\\text{PDFs}$"))
    pdfs_fig.show()


# Q5 Likelihood evaluation
def _evaluate_log_likelihood_heatmap(multivariate_gaussian, sigma, samples):
    ls_num_samples = 200
    f = np.linspace(-10, 10, ls_num_samples)
    log_likelihood = np.zeros((ls_num_samples, ls_num_samples))
    for i, f1 in enumerate(f):
        for j, f3 in enumerate(f):
            new_mu = np.array([f1, 0, f3, 0]).T
            ll = multivariate_gaussian.log_likelihood(new_mu, sigma, samples)
            log_likelihood[i, j] = ll

    ll_heatmap = go.Figure()
    ll_heatmap.add_heatmap(x=f, y=f, z=log_likelihood)
    ll_heatmap.update_layout(dict(title="Log Likelihood Heatmap",
                                  xaxis_title='r$f3\\text{ - 3rd }\mu\\text{ coordinate}$',
                                  yaxis_title='r$f1\\text{ - 1st }\mu\\text{ coordinate}$'))
    ll_heatmap.show()
    return f, log_likelihood


# Q6 Maximum likelihood
def _find_max_likelihood(f, log_likelihood):
    max_ll_indexes = np.unravel_index(np.argmax(log_likelihood, axis=None), log_likelihood.shape)
    max_ll = np.max(log_likelihood)
    print(f"Values of f1 and f3 that give maximum log-likelihood of {'%.3f' % max_ll} are: "
          f"{'%.3f' % f[max_ll_indexes[0]], '%.3f' % f[max_ll_indexes[1]]}")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    num_samples = 1000
    samples = np.random.multivariate_normal(mu, sigma, num_samples)

    multivariate_gaussian = MultivariateGaussian()
    fitted = multivariate_gaussian.fit(samples)
    print(f"{fitted.mu_}\n{fitted.cov_}")

    # Question 5 - Likelihood evaluation

    f, log_likelihood = _evaluate_log_likelihood_heatmap(multivariate_gaussian, sigma, samples)

    # Question 6 - Maximum likelihood

    _find_max_likelihood(f, log_likelihood)




if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
