import utils

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

# TEXT_EXPECTATION_DIFF_FUNC_OF_SAMPLES = "Estimation of Expectation Diff As Function Of Number Of Samples"
default_plot_height = 400

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()

    mu = 10
    sigma = 1
    num_samples = 1000
    samples = np.random.normal(mu, sigma, num_samples)
    univariate_gaussian = UnivariateGaussian()
    fitted = univariate_gaussian.fit(samples)
    print(f"({fitted.mu_}, {fitted.var_})")
    # print(f"pdfs: {univariate_gaussian.pdf(samples)}")
    # print(f"loglikelihood: {univariate_gaussian.log_likelihood(mu, sigma, samples)}")

    # Question 2 - Empirically showing sample mean is consistent
    # raise NotImplementedError()

    ms = np.linspace(10, 1000, 100, dtype=int)
    expectation_error = []
    for m in ms:
        estimated = UnivariateGaussian().fit(samples[:m])
        # print(f"estimated: ({estimated.mu_}, {estimated.var_})")
        expectation_error.append(abs(estimated.mu_ - mu))
    # go.Figure([go.Scatter(x=ms, y=estimated_distance, mode='markers+lines', name=r'$|\mu-\widehat\mu|$'),],
    expectation_error_fig = go.Figure([go.Scatter(x=ms, y=expectation_error, mode='lines')],
              layout=go.Layout(title=r"$\text{Expectation Error As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$|\mu-\hat\mu|\\text{ - expectation error}$",
                               height=default_plot_height))
    expectation_error_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()

    pdfs = univariate_gaussian.pdf(samples)
    pdfs_fig = go.Figure([go.Scatter(x=samples, y=pdfs, mode="markers", marker=dict(color="red"),
                          # name="$\\mathcal{N}\\left({mu},{sigma}\\right)$"
                          )],
              layout=go.Layout(title=r"$\text{Empirical PDF Function Under The Fitted Model}$",
                               xaxis_title="$x\\text{ - sample values}$",
                               yaxis_title="$\\text{PDFs}$",
                               height=default_plot_height))
    pdfs_fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    mu = [0, 0, 4, 0]
    sigma = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    num_samples = 1000
    samples = np.random.multivariate_normal(mu, sigma, num_samples)
    # samples = np.array([[150,45],[170,74],[184,79]])
    # mu = [168,66]

    multivariate_gaussian = MultivariateGaussian()
    fitted = multivariate_gaussian.fit(samples)
    print(f"{fitted.mu_}\n{fitted.cov_}")
    print(f"pdfs: {multivariate_gaussian.pdf(samples)}")

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
