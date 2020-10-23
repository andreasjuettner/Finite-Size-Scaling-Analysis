from frequentist_run import run_frequentist_analysis
from model_definitions import *
from tqdm import tqdm
from bayesian_functions import *


h5_data_file = "../Finite-Size-Scaling-Analysis/h5data/Bindercrossings.h5"


def get_pvalues_central_fit(N):
  """
    This function will reproduce the p-value data for the central fit as
    defined in the publication

    INPUTS :
    --------
    N: int, rank of the SU(N) valued fields

    OUTPUTS:
    --------
    pvalues_N2: (2, ) list of arrays of floats, containing the pvalues for the
      N = 2 analysis. Each array is of the length of the number of GL_min cut
      values, and the corresponding p-value to each cut is recorded. The first
      array is for the Lambda_IR = g / (4 pi N) model, while the second is for
      Lambda_IR = 1 / L

    pvalues_N4: (2, ) list of arrays of floats, containing the pvalues for the
      N = 4 analysis. Each array is of the length of the number of GL_min cut
      values, and the corresponding p-value to each cut is recorded. The first
      array is for the Lambda_IR = g / (4 pi N) model, while the second is for
      Lambda_IR = 1 / L
  """
  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  if N == 2:
    N_s = [2]
    Bbar_s = [0.52, 0.53]
    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]

    x0 = [0, 0.5431, -0.03586, 1, 2 / 3] # EFT values

    model1 = model1_1a
    model2 = model2_1a
    param_names = ["alpha", "f0", "f1", "lambduh", "nu"]

  if N == 4:
    N_s = [4]
    Bbar_s = [0.42, 0.43]
    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]

    x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values

    model1 = model1_2a
    model2 = model2_2a
    param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]

  pvalues_1 = numpy.zeros(len(GL_mins))
  pvalues_2 = numpy.zeros(len(GL_mins))

  for i, GL_min in enumerate(GL_mins):
    pvalues_1[i], params1, dof = run_frequentist_analysis(h5_data_file, model1, N_s, g_s, L_s, Bbar_s, GL_min, GL_max, param_names, x0, run_bootstrap=False)
    pvalues_2[i], params2, dof = run_frequentist_analysis(h5_data_file, model2, N_s, g_s, L_s, Bbar_s, GL_min, GL_max, param_names, x0, run_bootstrap=False)

  pvalues = [pvalues_1, pvalues_2]

  return pvalues


def get_statistical_errors_central_fit(N):
  """
    This function gets the statistical error bands (and central fit values)
    for the model parameters, and the value of the critical mass at g=0.1 quoted
    quoted in the publication.

    INPUTS :
    --------
    N: int, rank of the SU(N) valued fields

    OUTPUTS :
    ---------
    results_N2: list of the form [params, params_std, m_c, m_c_error]
      where params are the parameter estimates of the central fit, params_std
      are the statistical errors on these estimates, found through bootstrap
      resampling, m_c is the critcal mass, and m_c_error is the statistical
      error on this

    results_N4: list of the form
      [params, params_std, m_c1, m_c1_error, m_c2, m_c2_error],
      where params are the parameter estimates of the central fit, params_std
      are the statistical errors on these estimates, found through bootstrap
      resampling, m_c1 is the critcal mass estimated using the first alpha value,
      and m_c_error is the statistical error on this. Similarly for m_c2 and
      m_c2_error for the second alpha value
  """
  if N == 2:
    model = model1_1a
    N = 2
    N_s = [N]
    Bbar_s = [0.52, 0.53]
    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]
    GL_min = 12.8
    GL_max = 76.8

    Bbar_1 = "0.520"
    Bbar_2 = "0.530"
    x0 = [0, 0.5431, -0.03586, 1, 2 / 3] # EFT values
    param_names = ["alpha", "f0", "f1", "lambduh", "nu"]

  if N == 4:
    model = model1_2a
    N = 4
    N_s = [N]
    Bbar_s = [0.42, 0.43]
    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]
    GL_min = 12.8
    GL_max = 76.8

    x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values
    param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]

  # Run once with the full dataset (no resampling)
  pvalue, params_central, dof = run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s, GL_min, GL_max, param_names, x0, run_bootstrap=False)

  # Run with all the bootstrap samples
  pvalue, params, dof, sigmas = run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s, GL_min, GL_max, param_names, x0)

  # Calculate the value of the non-perterbative critical mass for g = 0.1 and it's
  # statistical error
  g = 0.1
  m_c = mPT_1loop(g, N) + g ** 2 * (params_central[0] - params_central[-2] * K1(g, N))
  print(f"m_c = {m_c}")

  alphas = params[..., 0]
  lambduhs = params[..., -2]

  m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - lambduhs * K1(g, N))

  minimum_m = numpy.min(m_cs)
  maximum_m = numpy.max(m_cs)

  print(f"m_c_range = {[minimum_m, maximum_m]}")

  m_c_error = max(m_c - minimum_m, maximum_m - m_c)
  print(f"m_c_error = {m_c_error}")

  if N == 2:
    results = [params_central, numpy.std(params, axis=0), m_c, m_c_error]

  if N == 4:
    alphas2 = params[..., 1]

    m_c2 = mPT_1loop(g, N) + g ** 2 * (params_central[1] - params_central[-2] * K1(g, N))
    print(f"m_c2 = {m_c2}")

    m_c2s = mPT_1loop(g, N) + g ** 2 * (alphas2 - lambduhs * K1(g, N))

    minimum_m = numpy.min(m_c2s)
    maximum_m = numpy.max(m_c2s)

    print(f"m_c2_range = {[minimum_m, maximum_m]}")

    m_c2_error = max(m_c2 - minimum_m, maximum_m - m_c2)
    print(f"m_c2_error = {m_c2_error}")

    results = [params_central, numpy.std(params, axis=0), m_c, m_c_error, m_c2, m_c2_error]

  return results


def get_systematic_errors(N):
  """
    This function gets the systematic error bands (and central fit values)
    for the model parameters, and the value of the critical mass at g=0.1 quoted
    quoted in the publication.

    INPUTS :
    --------
    N: int, rank of the SU(N) valued fields

    OUTPUTS :
    ---------
    results_N2: list of the form [params, params_std, m_c, m_c_error]
      where params are the parameter estimates of the central fit, params_std
      are the systematic errors on these estimates, found through bootstrap
      resampling, m_c is the critcal mass, and m_c_error is the systematic
      error on this

    results_N4: list of the form
      [params, params_std, m_c1, m_c1_error, m_c2, m_c2_error],
      where params are the parameter estimates of the central fit, params_std
      are the systematic errors on these estimates, found through bootstrap
      resampling, m_c1 is the critcal mass estimated using the first alpha value,
      and m_c_error is the systematic error on this. Similarly for m_c2 and
      m_c2_error for the second alpha value
  """
  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  if N == 2:
    model = model1_1a
    N_s = [2]
    Bbar_s = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]
    GL_min = 12.8
    GL_max = 76.8

    x0 = [0, 0.5431, -0.03586, 1, 2 / 3] # EFT values
    param_names = ["alpha", "f0", "f1", "lambduh", "nu"]
    min_dof = 15 # The minimum number of degrees of freedom needed to consider a fit valid

  if N == 4:
    model = model1_2a
    N_s = [4]
    Bbar_s = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47]
    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]
    GL_min = 12.8
    GL_max = 76.8

    x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values
    param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]
    min_dof = 15 # The minimum number of degrees of freedom needed to consider a fit valid

  n_params = len(param_names)

  # Make a list of all Bbar pairs
  Bbar_list = []
  for i in range(len(Bbar_s)):
    for j in range(i + 1, len(Bbar_s)):
      Bbar_list.append([Bbar_s[i], Bbar_s[j]])

  pvalues = numpy.zeros((len(Bbar_list), len(GL_mins)))
  params = numpy.zeros((len(Bbar_list), len(GL_mins), n_params))
  dofs = numpy.zeros((len(Bbar_list), len(GL_mins)))

  for i, Bbar_s in enumerate(Bbar_list):
    Bbar_1, Bbar_2 = Bbar_s
    print(f"Running fits with Bbar_1 = {Bbar_1}, Bbar_2 = {Bbar_2}")

    for j, GL_min in enumerate(GL_mins):
      pvalues[i, j], params[i, j], dofs[i, j] = run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s, GL_min, GL_max, param_names, x0, run_bootstrap=False)

  # Extract the index of the smallest GL_min fit that has an acceptable p-value
  r = len(GL_mins)
  best = r - 1

  for i, GL_min in enumerate(GL_mins):
    if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
      best = r - 1 - i

  best_Bbar_index = numpy.argmax(pvalues[:, best])
  best_Bbar = Bbar_list[best_Bbar_index]

  print("BEST RESULT")
  print(f"Bbar_s = {best_Bbar}")
  print(f"GL_min = {GL_mins[best]}")
  print(f"pvalue : {pvalues[best_Bbar_index, best]}")
  print(f"dof : {dofs[best_Bbar_index, best]}")

  params_central = params[best_Bbar_index, best]

  # Find the parameter variation over acceptable fits
  acceptable = numpy.logical_and(
               numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                      dofs >= min_dof)

  # Find the most extreme values of the parameter estimates that are deemed acceptable
  for i, param in enumerate(param_names):
    param_small = params[..., i]
    minimum = numpy.min(param_small[acceptable])
    maximum = numpy.max(param_small[acceptable])

  sys_sigmas = numpy.zeros(n_params)

  for i, param in enumerate(param_names):
    # Define the systematic error bar by the largest deviation from the central
    # fit by an acceptable fit
    sys_sigmas[i] = max(maximum - params[best_Bbar_index, best, i], params[best_Bbar_index, best, i] - minimum)

    print(f"{param} = {param} +- {sys_sigmas[i]}")

  # Find the systematic variation in the critical mass
  g = 0.1
  m_c = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 0] - params[best_Bbar_index, best, -2] * K1(g, N))
  print(f"m_c = {m_c}")

  alphas = params[..., 0]
  lambduhs = params[..., -2]

  # Only include parameter estimates from those fits that are acceptable
  alphas = alphas[acceptable]
  lambduhs = lambduhs[acceptable]

  m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - lambduhs * K1(g, N))

  minimum_m = numpy.min(m_cs)
  maximum_m = numpy.max(m_cs)

  print(f"m_c_range = {[minimum_m, maximum_m]}")

  m_c_error = max(m_c - minimum_m, maximum_m - m_c)
  print(f"m_c_error = {m_c_error}")

  if N == 2:
    results = [params_central, sys_sigmas, m_c, m_c_error]

  if N == 4:
    alphas2 = params[..., 1]

    # Only include parameter estimates from those fits that are acceptable
    alphas2 = alphas2[acceptable]

    # Calculate using alpha2
    m_c2 = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 1] - params[best_Bbar_index, best, -2] * K1(g, N))
    print(f"m_c2 = {m_c2}")

    m_c2s = mPT_1loop(g, N) + g ** 2 * (alphas2 - lambduhs * K1(g, N))

    minimum_m = numpy.min(m_c2s)
    maximum_m = numpy.max(m_c2s)

    print(f"m_c2_range = {[minimum_m, maximum_m]}")

    m_c2_error = max(m_c2 - minimum_m, maximum_m - m_c2)
    print(f"m_c2_error = {m_c2_error}")

    results = [params_central, sys_sigmas, m_c, m_c_error, m_c2, m_c2_error]

  return results


## Part IV: Calculating the Bayes Factor for figure 4.
def get_Bayes_factors(N):
  """
    This function produces the Bayes Factors shown in the publication. Note that
    the accuracy of the MULTINEST algorithm improves with increased points

    INPUTS :
    --------
    N: int, rank of the SU(N) valued fields

    OUTPUTS :
    ---------
    Bayes_factors2: The log of the Bayes factors of the Lambda_IR = g / (4 pi N)
      model over the Lambda_IR = 1 / L model, for N = 2 data. This is an array
      of lenght equal to the number of GL_min cuts considered, with each element
      containin the log Bayes factor of the corresponding GL_min cut.

    Bayes_factors4: The log of the Bayes factors of the Lambda_IR = g / (4 pi N)
      model over the Lambda_IR = 1 / L model, for N = 4 data. This is an array
      of lenght equal to the number of GL_min cuts considered, with each element
      containin the log Bayes factor of the corresponding GL_min cut.
  """
  GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
  GL_max = 76.8

  # Where the output samples will be saved
  directory = "MULTINEST_samples/"

  # How many sample points to use in the MULTINEST algorithm
  points = 800

  # Use this to label different runs if you edit something
  tag = ""

  # Prior Name: To differentiate results which use different priors
  prior_name = "A"

  # For reproducability
  seed = 3475642

  if N == 2:
    model = model1_1a
    N = 2
    N_s_in = [N]
    Bbar_s_in = [0.52, 0.53]
    g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s_in = [8, 16, 32, 48, 64, 96, 128]
    GL_max = 76.8

    model1 = model1_1a
    model2 = model2_1a
    param_names = ["alpha", "f0", "f1", "lambduh", "nu"]

    alpha_range = [-0.1, 0.1]
    f0_range = [0, 1]
    f1_range = [-2, 2]
    lambduh_range = [0, 2]
    nu_range = [0, 2]
    prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]

  if N == 4:
    model = model1_2a
    N = 4
    N_s_in = [N]
    Bbar_s_in = [0.42, 0.43]
    g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s_in = [8, 16, 32, 48, 64, 96, 128]
    GL_max = 76.8
    x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] # EFT values

    model1 = model1_2a
    model2 = model2_2a
    param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]

    alpha_range1 = [-0.1, 0.1]
    alpha_range2 = [-0.1, 0.1]
    f0_range = [0, 1]
    f1_range = [-2, 2]
    lambduh_range = [0, 2]
    nu_range = [0, 2]
    prior_range = [alpha_range1, alpha_range2, f0_range, f1_range, lambduh_range, nu_range]

  n_params = len(prior_range)
  Bayes_factors = numpy.zeros(len(GL_mins))

  for i, GL_min in enumerate(GL_mins):
    samples, g_s, L_s, Bbar_s, m_s = load_h5_data(h5_data_file, N_s_in, g_s_in, L_s_in, Bbar_s_in, GL_min, GL_max)

    dof = g_s.shape[0] - n_params

    cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

    analysis1, best_fit1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, prior_name=prior_name, keep_GLmax=False,
                              return_analysis_small=True, seed=seed)

    analysis2, best_fit2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, prior_name=prior_name, keep_GLmax=False,
                              return_analysis_small=True, seed=seed)

    # This is the log of the Bayes factor equal to the difference in the
    # log-evidence's between the two models
    Bayes_factors[i] = analysis1[0] - analysis2[0]

  # Change log bases to log10 to match the plot in the publication
  Bayes_factors = Bayes_factors / numpy.log(10)

  return Bayes_factors


if __name__ == "__main__":
  pvalues_N2 = get_pvalues_central_fit(2)
  pvalues_N4 = get_pvalues_central_fit(4)
  statistical_N2 = get_statistical_errors_central_fit(2)
  statistical_N4 = get_statistical_errors_central_fit(4)
  systematic_N2 = get_systematic_errors(2)
  systematic_N4 = get_systematic_errors(4)
  Bayes_factors_N2 = get_Bayes_factors(2)
  Bayes_factors_N4 = get_Bayes_factors(4)
