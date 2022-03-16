import utils

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    mu, sigma, num_samples = 10, 1, 1000
    samples = np.random.normal(mu, sigma, num_samples)
    univariate_gaussian = UnivariateGaussian()
    fitted = univariate_gaussian.fit(samples)
    print(f"({fitted.mu_}, {fitted.var_})")
    # print(f"pdfs: {univariate_gaussian.pdf(samples)}")
    # print(f"loglikelihood: {univariate_gaussian.log_likelihood(mu, sigma, samples)}")

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 100, dtype=int)
    expectation_error = []
    for m in ms:
        estimated = UnivariateGaussian().fit(samples[:m])
        # print(f"estimated: ({estimated.mu_}, {estimated.var_})")
        expectation_error.append(abs(estimated.mu_ - mu))
    # go.Figure([go.Scatter(x=ms, y=estimated_distance, mode='markers+lines', name=r'$|\mu-\widehat\mu|$'),],
    expectation_error_fig = go.Figure(go.Scatter(x=ms, y=expectation_error, mode='lines'),
                                      layout=go.Layout(
                                          title=r"$\text{Expectation Error As Function Of Number Of Samples}$",
                                          xaxis_title="$m\\text{ - number of samples}$",
                                          yaxis_title="r$|\mu-\hat\mu|\\text{ - expectation error}$"))
    expectation_error_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdfs = univariate_gaussian.pdf(samples)
    pdfs_fig = go.Figure(go.Scatter(x=samples, y=pdfs, mode="markers", marker=dict(color="red", size=1.5)),
                         layout=go.Layout(title=r"$\text{Empirical PDF Function Under The Fitted Model}$",
                                          xaxis_title="$x\\text{ - sample values}$",
                                          yaxis_title="$\\text{PDFs}$"))
    pdfs_fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    num_samples = 1000
    samples = np.random.multivariate_normal(mu, sigma, num_samples)
    # samples = np.array([[150,45],[170,74],[184,79]])
    # sigma = np.array([[292,301],[301,337]])
    # mu = np.array([168,66])

    multivariate_gaussian = MultivariateGaussian()
    fitted = multivariate_gaussian.fit(samples)
    print(f"{fitted.mu_}\n{fitted.cov_}")
    # print(f"pdfs: {multivariate_gaussian.pdf(samples)}")
    # print(f"loglikelihood: {multivariate_gaussian.log_likelihood(mu, sigma, samples)}")

    # Question 5 - Likelihood evaluation

    ls_num_samples = 200
    f = np.linspace(-10, 10, ls_num_samples)
    log_likelihood = np.zeros((ls_num_samples, ls_num_samples))
    for i, f1 in enumerate(f):
        for j, f3 in enumerate(f):
            new_mu = np.array([f1, 0, f3, 0]).T
            ll = multivariate_gaussian.log_likelihood(new_mu, sigma, samples)
            log_likelihood[i, j] = ll
    # print(f"x: {x}\ny: {y}\nlen(z): {len(z), len(z[0])}\n")

    # print(f"reallll: {multivariate_gaussian.log_likelihood(np.array([0, 0, 4, 0]), sigma, samples)}")
    fig = px.imshow(log_likelihood,
                    labels=dict(x='r$f3\\text{ - 3rd }\mu\\text{ coordinate}$',
                                y='r$f3\\text{ - 1st }\mu\\text{ coordinate}$', color='log-likelihood'),
                    x=f, y=f, origin='lower')
    fig.show()

    # Question 6 - Maximum likelihood

    max_vals = np.unravel_index(np.argmax(log_likelihood, axis=None), log_likelihood.shape)
    print(f"Values of f1 and f3 that give maximum likelihood are: "
          f"{'%.3f' % f[max_vals[0]], '%.3f' % f[max_vals[1]]}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
