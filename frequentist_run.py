from model_definitions import *
from scipy.optimize import minimize
from scipy.linalg import logm
from tqdm import tqdm


def run_frequentist_analysis(model, N, Bbar_1, Bbar_2, GL_min, GL_max, param_names, x0, method="BFGS", no_samples=500, run_bootstrap=True, print_info=True):
  samples, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl', GL_min=GL_min, GL_max=GL_max)

  cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

  # Using scipy.
  if method == "least_squares":
    res = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model), method=method)

  # Using scipy.optimize.minimize
  if method in ["dogbox", "Nelder-Mead", "Powell", "CG", "BFGS", "COBYLA"]:
    res = minimize(lambda x, y, z: numpy.sum(res_function(x, y, z) ** 2), x0, args=(cov_inv, model), method=method)

  chisq = chisq_calc(res.x, cov_inv, model, res_function)
  n_params = len(res.x)
  dof = g_s.shape[0] - n_params
  p = chisq_pvalue(dof, chisq)
  param_central = res.x

  if print_info:
    print(f"chisq = {chisq}")
    print(f"chisq/dof = {chisq / dof}")
    print(f"pvalue = {p}")
    print(f"dof = {dof}")

  # If the pvalue is acceptable, run a bootstrap to get a statistical error
  if p > 0.05 and run_bootstrap:
    param_estimates = numpy.zeros((no_samples, n_params))

    for i in tqdm(range(no_samples)):
      m_s = samples[:, i]
      
      res_function = make_res_function(N, m_s, g_s, L_s, Bbar_s)

      if method == "least_squares":
        res = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model), method=method)

      # Using scipy.optimize.minimize
      if method in ["dogbox", "Nelder-Mead", "Powell", "CG", "BFGS", "COBYLA"]:
        res = minimize(lambda x, y, z: numpy.sum(res_function(x, y, z) ** 2), x0, args=(cov_inv, model), method=method)

      param_estimates[i] = numpy.array(res.x)

    sigmas = numpy.std(param_estimates, axis=0)

    for i, param in enumerate(param_names):
      print(f"{param} = {param_central[i]} +- {sigmas[i]}")

    return p, res.x, dof, sigmas

  return p, res.x, dof


def run_frequentist_analysis(input_h5_file, model, N_s, g_s, L_s, Bbar_s, GL_min, GL_max, param_names, x0, method="BFGS", no_samples=500, run_bootstrap=True, print_info=True):
  samples, g_s, L_s, Bbar_s, m_s = load_h5_data(input_h5_file, N_s, g_s, L_s, Bbar_s, GL_min, GL_max)

  cov_matrix, different_ensemble = cov_matrix_calc(g_s, L_s, m_s, samples)
  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  res_function = make_res_function(N_s[0], m_s, g_s, L_s, Bbar_s)

  # Using scipy.
  if method == "least_squares":
    res = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model), method=method)

  # Using scipy.optimize.minimize
  if method in ["dogbox", "Nelder-Mead", "Powell", "CG", "BFGS", "COBYLA"]:
    res = minimize(lambda x, y, z: numpy.sum(res_function(x, y, z) ** 2), x0, args=(cov_inv, model), method=method)

  chisq = chisq_calc(res.x, cov_inv, model, res_function)
  n_params = len(res.x)
  dof = g_s.shape[0] - n_params
  p = chisq_pvalue(dof, chisq)
  param_central = res.x

  if print_info:
    print(f"chisq = {chisq}")
    print(f"chisq/dof = {chisq / dof}")
    print(f"pvalue = {p}")
    print(f"dof = {dof}")

  # If the pvalue is acceptable, run a bootstrap to get a statistical error
  if p > 0.05 and run_bootstrap:
    param_estimates = numpy.zeros((no_samples, n_params))

    for i in tqdm(range(no_samples)):
      m_s = samples[:, i]
      
      res_function = make_res_function(N_s[0], m_s, g_s, L_s, Bbar_s)

      if method == "least_squares":
        res = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model), method=method)

      # Using scipy.optimize.minimize
      if method in ["dogbox", "Nelder-Mead", "Powell", "CG", "BFGS", "COBYLA"]:
        res = minimize(lambda x, y, z: numpy.sum(res_function(x, y, z) ** 2), x0, args=(cov_inv, model), method=method)

      param_estimates[i] = numpy.array(res.x)

    sigmas = numpy.std(param_estimates, axis=0)

    for i, param in enumerate(param_names):
      print(f"{param} = {param_central[i]} +- {sigmas[i]}")

    return p, param_estimates, dof, sigmas

  return p, res.x, dof


if __name__ == "__main__":
  # Print out the result for the critical mass
  if model.__name__ == "NC_logg":
    g = 0.1
    m_c = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 0] - params[best_Bbar_index, best, -2] * K1(g, N))
    print(f"m_c = {m_c}")

    alphas = params[..., 0]
    lambduhs = params[..., -2]
    alphas = alphas[acceptable]
    lambduhs = lambduhs[acceptable]

    m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - lambduhs * K1(g, N))

    minimum_m = numpy.min(m_cs)
    maximum_m = numpy.max(m_cs)

    print(f"m_c_range = {[minimum_m, maximum_m]}")
    print(f"m_c_error = {max(m_c - minimum_m, maximum_m - m_c)}")


  if model.__name__ == "NC_logg_2a":
    g = 0.1
    m_c = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 0] - params[best_Bbar_index, best, -2] * K1(g, N))
    print(f"m_c = {m_c}")

    alphas = params[..., 0]
    lambduhs = params[..., -2]
    alphas = alphas[acceptable]
    lambduhs = lambduhs[acceptable]

    m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - lambduhs * K1(g, N))

    minimum_m = numpy.min(m_cs)
    maximum_m = numpy.max(m_cs)

    print(f"m_c_range = {[minimum_m, maximum_m]}")
    print(f"m_c_error = {max(m_c - minimum_m, maximum_m - m_c)}")

    g = 0.1
    m_c = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 1] - params[best_Bbar_index, best, -2] * K1(g, N))
    print(f"m_c = {m_c}")

    alphas = params[..., 1]
    lambduhs = params[..., -2]
    alphas = alphas[acceptable]
    lambduhs = lambduhs[acceptable]

    m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - lambduhs * K1(g, N))

    minimum_m = numpy.min(m_cs)
    maximum_m = numpy.max(m_cs)

    print(f"m_c_range = {[minimum_m, maximum_m]}")
    print(f"m_c_error = {max(m_c - minimum_m, maximum_m - m_c)}")
