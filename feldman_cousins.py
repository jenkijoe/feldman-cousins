import numpy as np
import matplotlib.pyplot as plt
import math as math
 
def _likelihood_ratio_gaussian(x, mu):
    if (x < 0):
        return math.exp(x * mu - mu * mu/2)
    else: 
        return math.exp(-1 * (x - mu)*(x-mu) / 2)
 
def _likelihood_ratio_poisson(lmda, n):
    if (n <= 0):
        return math.exp(-lmda)
    else:   
        return (lmda**n) * math.exp(-lmda) / math.factorial(n)

def feldman_cousins_gaussian(mu_min, mu_max, step_size, sigma, trials=1000000 , confidence_limit=0.9):
    """
    Calculate Feldman Cousins confidence intervals for toy gaussian bound by mu > 0

    Args:
        mu_min -- lower limit for mu parameter scan
        max_max -- upper limit for mu parameter scan
        step_size -- step_size for mu parameter scan
        sigma -- standard deviation for random number gaussian distribution
        trials --  number of Monte-Carlo trials (default: 1000000)
        confidence_limit -- required confidence limit in the range [0, 1.0) (default: 0.9 for 90% CL)
    
    Returns:
        numpy array representation for mu confidence belt: [0] = mu, [1] is lower confidence limit, [2] is upper confidence limit

    """

    #initialise arrays
    mu_values = np.arange(mu_min + step_size, mu_max, step_size)
    results = np.zeros([3, mu_values.shape[0]], dtype=float)
 
    index = 0
    for mu in mu_values:
        
        x_values = np.random.normal(mu, sigma, trials)
        likelihood_ratios = map(_likelihood_ratio_gaussian, x_values, np.ones(trials) * mu)
 
        # sort them by r
        idx = np.argsort(likelihood_ratios, 0)[::-1]
        sorted_r = [likelihood_ratios[i] for i in idx]
        sorted_x = [x_values[i] for i in idx]
     
        # produce list of indices 
        cut_off = int(math.floor(confidence_limit * trials))
        accepted = sorted_x[0:cut_off]
 
        #get min and max x values from the set of remaining x values
        results[0][index] = mu
        results[1][index] = min(accepted)
        results[2][index] = max(accepted)
        index = index+1
    return results

def feldman_cousins_poisson(mu_min, mu_max, step_size, n_background, trials=1000000, confidence_limit=0.9):

    n_max = int(mu_max)
    
    mu_values = np.arange(mu_min, mu_max, step_size)
    n_values = np.arange(n_max)
    mu_best = np.array(map(max, np.zeros(n_max), n_values-n_background))
    results = np.zeros([3,mu_values.shape[0]], dtype=float) # [0] = mu, [1] is lower confidence limit, [2] is upper confidence limit
 
    index = 0
    for mu in mu_values:
        #calculate probability for each n
        prob_values = np.array(map(_likelihood_ratio_poisson, np.ones(n_max)*(mu+n_background), n_values))
        prob_best = np.array(map(_likelihood_ratio_poisson, mu_best+n_background, n_values))
        likelihood_ratios = np.divide(prob_values, prob_best)

        #sort the likelihood ratios
        idx = np.argsort(likelihood_ratios, 0)[::-1]
        sorted_n = [n_values[i] for i in idx]
        sorted_p = [prob_values[i] for i in idx]
    
        sum_p = 0.0
        cut_off = 0
        for p in sorted_p:
            sum_p = sum_p + p
            cut_off = cut_off + 1
            if (sum_p > confidence_limit):
                break
         
        accepted = sorted_n[0:cut_off]
     
        results[0][index] = mu
        results[1][index] = min(accepted)
        results[2][index] = max(accepted)
        index = index + 1
    return results

def _plot(results, x_limits, y_limits, axis_names):
    plt.plot(results[1], results[0])
    plt.plot(results[2], results[0])
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.show()

def main():

    # Example 1: toy gaussian bound by mu > 0 
    results_poisson = feldman_cousins_poisson(0.0, 30.0, 0.001, 3.0, 1000000, 0.9)    
    #_plot(results_poisson, (0.0, 15.0), (0.0, 15.0), ("Measured n", "Measured mu")) 
    
    # Example 2: toy poisson with background count of 3.0
    results_gaussian = feldman_cousins_gaussian(0.0, 6.0, 0.05, 1.0, 1000000, 0.9)
    #_plot(results_gaussian, (-4.0, 6.0), (0.0, 6.0), ("Measured x", "Mean mu"))

if __name__ == '__main__':
    main()


