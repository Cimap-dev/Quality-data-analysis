import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from statsmodels.tsa.stattools import acf
import statsmodels.graphics.tsaplots as sgt
import seaborn as sns
import statsmodels.api as sm
import qda
from statsmodels.sandbox.stats.runs import runstest_1samp
from sklearn.decomposition import PCA

def normploting1000samples(mu,sigma):
  
    y = np.random.normal(mu, sigma, 1000) #y is a series of 1000 values

    # Create a histogram
    plt.hist(y)

    # Show the plot
    plt.show()

    return(y)

def chi2_test(var_data,var_H0,dof,direction='bilateral',alpha=0.05):
    chi2stat=dof*(var_data/var_H0) #(n-1)*S^2
    print('the value of the test statistic (n-1)*(S^2/sigma0^2) is %.3f' %chi2stat)
    if(direction=='bilateral'):
        critical_value_L=stats.chi2.ppf(alpha/2,dof)
        critical_value_U=stats.chi2.ppf(1-alpha/2,dof)
        critical_values=[critical_value_L,critical_value_U]
        p_value = 2 * min(stats.chi2.cdf(chi2stat, dof), 1 - stats.chi2.cdf(chi2stat, dof))
        print('critical values are %.3f and %.3f' %(critical_values[0],critical_values[1]))
        print('')
        print('p_value is %.3f'%p_value)
        print('')
        if(p_value>alpha):
            print('we have to accept')
        else:
            print('we have to reject')
    elif(direction=='greater'):
        critical_value_U=stats.chi2.ppf(1-alpha,dof)
        critical_values=critical_value_U
        p_value=1-stats.chi2.cdf(chi2stat,dof)
        print('critical value is %.3f' %critical_values)
        print('')
        print('p_value is %.3f'%p_value)
        print('')
        if(p_value>alpha):
            print('we have to accept')
        else:
            print('we have to reject')
    elif(direction=='lower'):
        critical_value_L=stats.chi2.ppf(alpha,dof)
        critical_values=critical_value_L
        p_value=stats.chi2.cdf(chi2stat,dof)
        print('critical value is %.3f' %critical_values)
        print('')
        print('p_value is %.3f'%p_value)
        print('')
        if(p_value>alpha):
            print('we have to accept')
        else:
            print('we have to reject')   
    return chi2stat,critical_values,p_value

def power_Z_single_plot (n, delta, alpha, mu0, sigma, sided, mu1_greater, plot_beta='no'):
    standard_error_mean=sigma/np.sqrt(n)
    if (mu1_greater == 'true'):
        mu1 = mu0 + delta #valore centrale della distribuzione sotto H1
    elif (mu1_greater == 'false'):
        mu1 = mu0 - delta 

    if (sided=='no'):
        Z_alpha2 = stats.norm.ppf(1 - alpha / 2) #valore critico, alpha è fissato (se no varia anche con alpha)
        power= 1 - stats.norm.cdf(Z_alpha2 - delta/standard_error_mean) + stats.norm.cdf(-Z_alpha2 - delta/standard_error_mean)
    elif (sided=='upper'):
        Z_alpha= stats.norm.ppf(1 - alpha) #valore critico, alpha è fissato (se no varia anche con alpha)
        power= 1 - stats.norm.cdf(Z_alpha - delta/standard_error_mean)
    elif (sided=='lower'):
        Z_alpha= stats.norm.ppf(1 - alpha) #valore critico, alpha è fissato (se no varia anche con alpha)
        power= stats.norm.cdf(-Z_alpha + delta/standard_error_mean)

    beta=1-power

    # Plot the power curve
    if plot_beta!='no':
        plt.plot(delta, power)
        plt.xlabel("delta")
        plt.ylabel("power")
        plt.grid(True)
        plt.show()
    else:
        plt.plot(delta, beta)
        plt.xlabel("delta")
        plt.ylabel("beta")
        plt.title('Operating characteristic curve')
        plt.grid(True)
        plt.show()

def single_ts_plot(data, x_label='x', y_label='y'):
    n=len(data) #number of datapoint in the time series of sales
    print("Number of points n = %d" % n)

    mean = data.mean()
    print('Mean = %.2f'% mean) #mean of the points

    # Let's plot the data first
    plt.figure(figsize=(15, 5))
    plt.plot(data, 'o-')
    plt.hlines(mean, 0, n, colors='r', linestyles='dashed') #0 è da dove parte sull'asse x, è l'x-min del grafico
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()
    plt.show()

def multiple_ts_plot(data, columns=None, title="Time Series Plot", xlabel="Time", ylabel="Value"):
    """
    Plots time series for specified columns in a dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the time series data.
    - columns (list, optional): List of column names to plot. If None, all columns will be plotted.
    - title (str, optional): Title of the plot.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    """
    if columns is None:
        columns = data.columns
    
    plt.figure(figsize=(12, 6))
    for column in columns:
        if column in data.columns:
            plt.plot(data.index, data[column], label=column)
        else:
            print(f"Column '{column}' not found in the dataframe.")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.show()

def histandbox(data, multiobservation='no'):
    if multiobservation!='no':
       data_stack=data.stack() 
    else:
        data_stack=data
    plt.hist(data_stack)
    plt.title('Histogram')
    plt.show()

    plt.boxplot(data_stack)
    plt.title('Boxplot')
    plt.show()

def shapiroqq (data, alpha=0.05, multiobservation='no'):
    if multiobservation!='no':
       data_stack=data.stack() 
    else:
        data_stack=data.copy()
    data_stack=data_stack.dropna()
    _, p_value_SW = stats.shapiro(data_stack)
    print('p-value of the Shapiro-Wilk test: %.5f' % p_value_SW)

    plt.show()
    if p_value_SW < alpha:
        print('Reject H0: the data are not normal')
    else:
        print('Accept H0: the data are normal')
    
        # QQ-plot
    stats.probplot(data_stack, dist='norm', plot=plt)

def boxcox(data, nomecolonnadati,approx_to_0_if_nec='False',simplify='False',add_cost=0):
    [data_norm, lmbda] = stats.boxcox(data[nomecolonnadati]+add_cost) #otteniamo i dati normalizzati e il lambda ottimo stimato
    print('il Lambda migliore è= %.4f' % lmbda)

    if (lmbda < 0.10 and lmbda>-0.10):
        if(approx_to_0_if_nec!='False'): #se sono nel range e se approssimo a 0
            lmbda=0
            print('ma il lambda usato è 0')
            data_norm = (data[nomecolonnadati]+add_cost).transform(lambda x: ((np.log(x))))
    else: #se invece lambda non rientra in quel range allora decido se modificare o meno:
        if(simplify!='False'): #in questo caso modifico
            data_norm = (data[nomecolonnadati]+add_cost).transform(lambda x: (x**lmbda))

    _, p_value_SW = stats.shapiro(data_norm)
    print('p-value of the Shapiro-Wilk test: %.5f' % p_value_SW)
    if (p_value_SW>0.05):
        print('box cox succeeded :)')
    plt.hist(data_norm)
    plt.title('Histogram of Box-Cox transformed data')
    plt.show()
    return data_norm,lmbda

def runs(data, alpha=0.05):
    
    stat, pval_runs = runstest_1samp(data, correction=False)
    print('Runs test statistic = {:.3f}'.format(stat))
    print('Runs test p-value = {:.5f}'.format(pval_runs))

    if pval_runs < alpha:
        print('Reject H0: the data are not random')
    else:
        print('Accept H0: the data are random')
 

def bartlett_test (data_norm, lag_test, alpha=0.05, salta=0): #data_norm can be residuals
        
    n = len(data_norm[salta:])
    #autocorrelation function
    [acf_values, lbq, _] = acf(data_norm[salta:], nlags = int(np.sqrt(n)), qstat=True, fft = False)
    rk = acf_values[lag_test]
    z_alpha2 = stats.norm.ppf(1-alpha/2)
    p_value = 2 * (1 - stats.norm.cdf(abs(rk * np.sqrt(n)))) #sistemato
    print('Standardized Test statistic rk*sqrt(n) = %f' %(rk*np.sqrt(n)))
    print('Rejection region of the standard normal starts at %f' % (z_alpha2))
    print('p-value = %f' % p_value)

    if abs(rk)>(z_alpha2/np.sqrt(n)):
        print('The null hypothesis is rejected')
    else: print('The null hypothesis is accepted')

def pred_interval_DM (data_norm, alpha=0.05): 
    df = len(data_norm) - 1
    Xbar = data_norm.mean()
    s = data_norm.std()
    t_alpha = stats.t.ppf(1 - alpha/2, df)

    [pred_lo, pred_up] = [Xbar-t_alpha*s,Xbar+t_alpha*s]
    print('Two-sided prediction interval for transformed data: [%.3f %.3f]' % (pred_lo, pred_up))

    return  [pred_lo, pred_up] 

def gapping(data, gap_size):
    gap_num= int(len(data)/gap_size)
    gap_data= np.zeros((gap_num))
    for i in range (gap_num):
        gap_data[i]=data[i*gap_size]

    plt.plot(gap_data, 'o-')
    plt.title('Time series plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

    return gap_data

def batching(data, column, batch_size):
    # Extract the column from the DataFrame
    values = data[column].values
    
    # Calculate the number of full batches
    num_full_batches = len(values) // batch_size
    
    # Initialize an empty list to store the batch averages
    batch_averages = []
    
    # Iterate through the full batches
    for i in range(num_full_batches):
        # Calculate the start and end index of the current batch
        start_index = i * batch_size
        end_index = start_index + batch_size
        
        # Compute the average of the current batch
        batch_avg = np.mean(values[start_index:end_index])
        
        # Append the batch average to the list
        batch_averages.append(batch_avg)

    plt.plot(batch_averages, 'o-')
    plt.title('Time series plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show() 
    
    return batch_averages


def scatter_plot_nice_h(data, nomecolonna1, nomecolonna2, highlight_index=None):
    """
    Plots a scatter plot of two columns from a dataframe, fits a regression line, 
    and optionally highlights a specific observation.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.
    nomecolonna1 (str): The column name for the x-axis.
    nomecolonna2 (str): The column name for the y-axis.
    highlight_index (int, optional): The index of the observation to highlight. Default is None.
    """

    sns.regplot(x=nomecolonna1, y=nomecolonna2, data=data, fit_reg=True, ci=None, 
                line_kws={'color': 'red', 'lw': 2, 'ls': '--'})
    
    if highlight_index is not None:
        if highlight_index in data.index:
            highlighted_point = data.loc[highlight_index]
            plt.scatter(highlighted_point[nomecolonna1], highlighted_point[nomecolonna2], 
                        color='red', s=100, edgecolor='k', label='Highlighted point')
    
    plt.title('Scatter plot of %s vs %s' % (nomecolonna1, nomecolonna2))
    plt.xlabel('%s' % nomecolonna1)
    plt.ylabel('%s' % nomecolonna2)
    plt.grid()
    plt.legend()
    plt.show()


def fitsARd1 (data,nomecolonnax, nomecolonnay,add_costant,lag_AR,d):
    x = data[nomecolonnax][lag_AR+d:] 
    if(add_costant=='true' or add_costant=='True' or add_costant=='yes') :
        x = sm.add_constant(x)
    y = data[nomecolonnay][lag_AR+d:]#idem
    model = sm.OLS(y, x).fit()
    qda.summary(model)

    return model

def shapirplusresplots(model,salta=0):
    residuals=model.resid[salta:]
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Residual Plots')

    axs[0,0].set_title('Normal probability plot')
    stats.probplot(residuals, dist="norm", plot=axs[0,0])

    axs[0,1].set_title('Versus Fits')
    axs[0,1].scatter(model.fittedvalues[salta:], residuals)

    fig.subplots_adjust(hspace=0.5)

    axs[1,0].set_title('Histogram')
    axs[1,0].hist(model.resid)

    axs[1,1].set_title('Time series plot')
    axs[1,1].plot(np.arange(1, len(residuals)+1), residuals, 'o-')

    _, pval_SW_res = stats.shapiro(residuals)
    print('Shapiro-Wilk test p-value on the residuals = %.5f' % pval_SW_res)

def ci_param_model(data,model, associated_regressor,number_of_parameters_of_the_model,alpha=0.05):
    betax = model.params[associated_regressor] 

    print('The estimated coefficient betax is %.3f' % betax)

    se_betax = model.bse[associated_regressor] #bse: beta standard error, di nuovo specifichiamo il lag
    print('The standard error of the estimated coefficient beta1 is %.3f' % se_betax)

    n = len(data)
    t_alpha2 = stats.t.ppf(1-alpha/2, n-number_of_parameters_of_the_model)

    CI_betax = [betax - t_alpha2*se_betax, betax + t_alpha2*se_betax]

    print('The confidence interval for betax is [%.3f, %.3f]' % (CI_betax[0], CI_betax[1]))

def ci_mean_model_ar1(data, model, lag_specifico, alpha=0.05):
    n=len(data)
    Xbar = data[lag_specifico].mean()          # sample mean of the regressor
    S2_X = data[lag_specifico].var()           # sample variance of the regressor

    p = len(model.model.exog_names)     # number of parameters
    S2_Y = np.var(model.resid, ddof=p)  # sample variance of residuals

    last_lag = data['Ex4'].iloc[-1] #nuova variable che contiene x35, l'ultimo item dell'array
    print('X_35 = %.3f' % last_lag)

    #predict the next value
    Yhat = model.predict([1,last_lag]) #dentro dobbiamo mettere 1 valore per regressore, 1 e last_lag
    print('Next process outcome = %.3f' % Yhat)

    t_alpha2 = stats.t.ppf(1-alpha/2, n-2)
    CI = [Yhat - t_alpha2*np.sqrt(S2_Y*(1/n + ((last_lag - Xbar)**2)/((n-1)*S2_X))),
            Yhat + t_alpha2*np.sqrt(S2_Y*(1/n + ((last_lag - Xbar)**2)/((n-1)*S2_X)))]
    print('The confidence interval for the mean response is [%.3f, %.3f]' % (CI[0], CI[1]))

def acfpacf(data):

    fig, ax = plt.subplots(2, 1) #è un subplot
    sgt.plot_acf(data, lags = int(len(data)/3), zero=False, ax=ax[0])
    fig.subplots_adjust(hspace=0.5)
    sgt.plot_pacf(data, lags = int(len(data)/3), zero=False, ax=ax[1], method = 'ywm')
    plt.show()

def diffprocess (data,first,second,order):
 
    diff= data[first]- data[second] #è come avere un nuovo dataset

    plt.plot(diff, 'o-')
    plt.xlabel('Index')
    plt.ylabel('DIFF%s' %order)
    plt.title('Time series plot of DIFF 1')
    plt.grid()
    plt.show()
    stat, pval_runs = runstest_1samp(diff[order:], correction=False)
    print('Runs test statistic = {:.5f}'.format(stat))
    print('Runs test p-value = {:.5f}'.format(pval_runs))
    alfa=0.05
    if pval_runs < alfa:
        print('Reject H0: the data are not random')
    else:
        print('Accept H0: the data are random')

    fig, ax = plt.subplots(2, 1)
    sgt.plot_acf(diff[order:], lags = int(len(data)/3), zero=False, ax=ax[0])
    fig.subplots_adjust(hspace=0.5)
    sgt.plot_pacf(diff[order:], lags = int(len(data)/3), zero=False, ax=ax[1], method = 'ywm')
    plt.show()
    return diff

def plotyvsfits(y,model):
    plt.plot(y, 'o-', label='Original data')
    plt.xlabel('Index') 
    plt.ylabel('y')
    plt.plot(model.fittedvalues, 's--', color='red', label='Fitted values', alpha=0.5)
    plt.legend()
    plt.grid()
    plt.show()

def scatter_b_vars(data):
    pd.plotting.scatter_matrix(data, alpha = 1)
    plt.show()

def compute_eigens(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    print(eigenvalues)
    print(eigenvectors)
    print(cumulative_explained_variance_ratio)
    return eigenvalues, eigenvectors, cumulative_explained_variance_ratio


def project_data_pca(data, mu, cov_matrix, eigenvectors,components_to_use, standardize='no'):
    # Extract standard deviations from the covariance matrix (sqrt of the diagonal elements)
    std_devs = np.sqrt(np.diag(cov_matrix)) 
    # Center the dataset (subtract the mean of each feature)
    X_centered = data.sub(mu.values)
    # Standardize the dataset (divide by the standard deviation of each feature)
    if standardize!='no':
        X_centered = X_centered / std_devs
    # Project the standardized data

    eigenvectors = eigenvectors[:, :components_to_use]
    X_projected = np.dot(X_centered, eigenvectors)
    pc_columns = [f'PC_{i+1}' for i in range(X_projected.shape[1])]
    X_projected_df = pd.DataFrame(X_projected, columns=pc_columns)
    print(X_projected_df.head())
    return X_projected_df

def p_pca(data, sample_to_use, standardize='no'):
    data_to_use=data.iloc[:sample_to_use,:]
    p=len(data.columns)
    data_centered_to_use = data_to_use - data_to_use.mean()
    data_centered=data-data_to_use.mean()
    data_std_to_use = data_centered_to_use / data_to_use.std()
    data_std=data_centered/data_to_use.std()
    if standardize!='no':
        data_centered_to_use=data_std_to_use
        data_centered=data_std
    pca = PCA()
    # Fit the PCA object to the data
    pca.fit(data_centered_to_use)
    # Print the eigenvalues
    print("Eigenvalues \n", pca.explained_variance_)
    # Print the eigenvectors 
    print("\nEigenvectors aka Loadings \n", pca.components_)
    # Print the explained variance ratio
    print("\nExplained variance ratio \n", pca.explained_variance_ratio_)
    # Print the cumulative explained variance ratio
    print("\nCumulative explained variance ratio \n", np.cumsum(pca.explained_variance_ratio_))
    
    scores = pca.transform(data_centered) #it's a numpy array n x p
   
    columnss = [f'z{i+1}' for i in range(p)]

    # Create the DataFrame with dynamic column names
    scores_df = pd.DataFrame(scores, columns=columnss)
   
    # Print the first rows of the scores dataframe
    print('first 5 rows of the df scores:')
    print(scores_df.head())

    return pca, scores_df

def plotloadings(pca):
    p=len(pca.components_)
    fig, ax = plt.subplots(1, p, figsize = (15, 5))
    #first graph:
    for i in range(1,p+1):
        ax[i-1].plot(pca.components_[i-1], 'o-')
        ax[i-1].set_title('Loading %d'%i)
    #show:
    plt.show()

def screeplotandcumexplvar(pca):
    # Plot the eigenvalues (scree plot)
    plt.plot(pca.explained_variance_, 'o-')
    plt.xlabel('Component number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree plot')
    plt.show()

    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    # add a bar chart to the plot
    plt.bar(range(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, width = 0.5, alpha=0.5, align='center')
    plt.xlabel('Component number')
    plt.ylabel('Cumulative explained variance')
    plt.title('Cumulative explained variance')
    plt.show()

def xbars_mu_given (mu,std_or_sigma,n,alpha,sample_mean, sample_std,subset_size=None):


    K=stats.norm.ppf(1-alpha/2)
    c4=qda.constants.getc4(n)

    data_XS = pd.DataFrame(columns=['Xbar_CL', 'Xbar_UCL', 'Xbar_LCL','S_CL','S_UCL','S_LCL'])

    # Compute the CL, UCL and LCL for Xbar and S
    Xbar_CL = mu
    Xbar_UCL = mu + K*std_or_sigma/np.sqrt(n)
    Xbar_LCL = mu - K*std_or_sigma/np.sqrt(n)

    S_CL = c4 * std_or_sigma  # Expected value of s (sample standard deviation)
    S_UCL = c4 * std_or_sigma + K * np.sqrt(1 - c4**2) * std_or_sigma
    S_LCL = c4 * std_or_sigma - K * np.sqrt(1 - c4**2) * std_or_sigma
    if S_LCL < 0:
        S_LCL=0

    for i in range(len(sample_mean)):
        data_XS.loc[i] = [Xbar_CL, Xbar_UCL, Xbar_LCL, S_CL, S_UCL, S_LCL]
    data_XS['sample_mean']=sample_mean
    data_XS['sample_std']=sample_std

    data_XS['Xbar_TEST1'] = np.where((data_XS['sample_mean'] > data_XS['Xbar_UCL']) | 
                (data_XS['sample_mean'] < data_XS['Xbar_LCL']), data_XS['sample_mean'], np.nan)
    data_XS['S_TEST1'] = np.where((data_XS['sample_std'] > data_XS['S_UCL']) | 
                    (data_XS['sample_std'] < data_XS['S_LCL']), data_XS['sample_std'], np.nan)

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle(('Xbar-S charts'))
    ax[0].plot(data_XS['sample_mean'], color='mediumblue', linestyle='--', marker='o')
    ax[0].plot(data_XS['Xbar_UCL'], color='firebrick', linewidth=1)
    ax[0].plot(data_XS['Xbar_CL'], color='g', linewidth=1)
    ax[0].plot(data_XS['Xbar_LCL'], color='firebrick', linewidth=1)
    ax[0].set_ylabel('Sample Mean')
        # add the values of the control limits on the right side of the plot
    ax[0].text(len(data_XS)+.5, data_XS['Xbar_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XS['Xbar_UCL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XS)+.5, data_XS['Xbar_CL'].iloc[0], 'CL = {:.3f}'.format(data_XS['Xbar_CL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XS)+.5, data_XS['Xbar_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XS['Xbar_LCL'].iloc[0]), verticalalignment='center')
        # highlight the points that violate the alarm rules
    ax[0].plot(data_XS['Xbar_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

    ax[1].plot(data_XS['sample_std'], color='mediumblue', linestyle='--', marker='o')
    ax[1].plot(data_XS['S_UCL'], color='firebrick', linewidth=1)
    ax[1].plot(data_XS['S_CL'], color='g', linewidth=1)
    ax[1].plot(data_XS['S_LCL'], color='firebrick', linewidth=1)
    ax[1].set_ylabel('Sample StDev')
    ax[1].set_xlabel('Sample Number')
        # add the values of the control limits on the right side of the plot
    ax[1].text(len(data_XS)+.5, data_XS['S_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XS['S_UCL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(data_XS)+.5, data_XS['S_CL'].iloc[0], 'CL = {:.3f}'.format(data_XS['S_CL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(data_XS)+.5, data_XS['S_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XS['S_LCL'].iloc[0]), verticalalignment='center')
        # highlight the points that violate the alarm rules
    ax[1].plot(data_XS['S_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
        # set the x-axis limits
    ax[1].set_xlim(-1, len(data_XS))

    if subset_size!=None:
            ax[0].axvline(x=subset_size-.5, color='k', linestyle='--')
            ax[1].axvline(x=subset_size-.5, color='k', linestyle='--')
    return data_XS

def XbarS_prob(mu_or_mean,std_or_sigma,n,alpha,sample_mean, sample_std, subset_size=None):


    K=stats.norm.ppf(1-alpha/2)
    L=stats.chi2.ppf(alpha/2,n-1)
    U=stats.chi2.ppf(1-alpha/2,n-1)
    data_XS = pd.DataFrame(columns=['Xbar_CL', 'Xbar_UCL', 'Xbar_LCL','S_UCL','S_LCL'])

    # Compute the CL, UCL and LCL for Xbar and S
    Xbar_CL = mu_or_mean
    Xbar_UCL = mu_or_mean + K*std_or_sigma/np.sqrt(n)
    Xbar_LCL = mu_or_mean - K*std_or_sigma/np.sqrt(n)

    S_UCL = np.sqrt((U*std_or_sigma**2)/(n-1))
    S_LCL = np.sqrt((L*std_or_sigma**2)/(n-1))
    if S_LCL < 0:
        S_LCL=0

    for i in range(len(sample_mean)):
        data_XS.loc[i] = [Xbar_CL, Xbar_UCL, Xbar_LCL, S_UCL, S_LCL]
    data_XS['sample_mean']=sample_mean
    data_XS['sample_std']=sample_std

    data_XS['Xbar_TEST1'] = np.where((data_XS['sample_mean'] > data_XS['Xbar_UCL']) | 
                (data_XS['sample_mean'] < data_XS['Xbar_LCL']), data_XS['sample_mean'], np.nan)
    data_XS['S_TEST1'] = np.where((data_XS['sample_std'] > data_XS['S_UCL']) | 
                    (data_XS['sample_std'] < data_XS['S_LCL']), data_XS['sample_std'], np.nan)

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle(('Xbar-S charts'))
    ax[0].plot(data_XS['sample_mean'], color='mediumblue', linestyle='--', marker='o')
    ax[0].plot(data_XS['Xbar_UCL'], color='firebrick', linewidth=1)
    ax[0].plot(data_XS['Xbar_CL'], color='g', linewidth=1)
    ax[0].plot(data_XS['Xbar_LCL'], color='firebrick', linewidth=1)
    ax[0].set_ylabel('Sample Mean')
        # add the values of the control limits on the right side of the plot
    ax[0].text(len(data_XS)+.5, data_XS['Xbar_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XS['Xbar_UCL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XS)+.5, data_XS['Xbar_CL'].iloc[0], 'CL = {:.3f}'.format(data_XS['Xbar_CL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XS)+.5, data_XS['Xbar_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XS['Xbar_LCL'].iloc[0]), verticalalignment='center')
        # highlight the points that violate the alarm rules
    ax[0].plot(data_XS['Xbar_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

    ax[1].plot(data_XS['sample_std'], color='mediumblue', linestyle='--', marker='o')
    ax[1].plot(data_XS['S_UCL'], color='firebrick', linewidth=1)
    ax[1].plot(data_XS['S_LCL'], color='firebrick', linewidth=1)
    ax[1].set_ylabel('Sample StDev')
    ax[1].set_xlabel('Sample Number')
        # add the values of the control limits on the right side of the plot
    ax[1].text(len(data_XS)+.5, data_XS['S_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XS['S_UCL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(data_XS)+.5, data_XS['S_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XS['S_LCL'].iloc[0]), verticalalignment='center')
        # highlight the points that violate the alarm rules
    ax[1].plot(data_XS['S_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
        # set the x-axis limits
    ax[1].set_xlim(-1, len(data_XS))

    if subset_size!=None:
            ax[0].axvline(x=subset_size-.5, color='k', linestyle='--')
            ax[1].axvline(x=subset_size-.5, color='k', linestyle='--')
    return data_XS


def xbarR(mean_or_mu,std_or_sigma,n,alpha,sample_mean, sample_R,subset_size=None):
    K=stats.norm.ppf(1-alpha/2)
    d2=qda.constants.getd2(n)
    d3=qda.constants.getd3(n)
    data_XR = pd.DataFrame(columns=['Xbar_CL', 'Xbar_UCL', 'Xbar_LCL','R_CL','R_UCL','R_LCL'])

    # Compute the CL, UCL and LCL for Xbar and S
    Xbar_CL = mean_or_mu
    Xbar_UCL = mean_or_mu + K*std_or_sigma/np.sqrt(n)
    Xbar_LCL = mean_or_mu - K*std_or_sigma/np.sqrt(n)

    R_CL = d2 * std_or_sigma  # Expected value of s (sample standard deviation)
    R_UCL = d2 * std_or_sigma + K * d3 * std_or_sigma
    R_LCL = d2 * std_or_sigma - K * d3 * std_or_sigma
    if R_LCL < 0:
        R_LCL=0

    for i in range(len(sample_mean)):
        data_XR.loc[i] = [Xbar_CL, Xbar_UCL, Xbar_LCL, R_CL, R_UCL, R_LCL]
    data_XR['sample_mean']=sample_mean
    data_XR['sample_range']=sample_R

    data_XR['Xbar_TEST1'] = np.where((data_XR['sample_mean'] > data_XR['Xbar_UCL']) | 
                    (data_XR['sample_mean'] < data_XR['Xbar_LCL']), data_XR['sample_mean'], np.nan)
    data_XR['R_TEST1'] = np.where((data_XR['sample_range'] > data_XR['R_UCL']) | 
                    (data_XR['sample_range'] < data_XR['R_LCL']), data_XR['sample_range'], np.nan)

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle(('Xbar-R charts'))
    ax[0].plot(data_XR['sample_mean'], color='mediumblue', linestyle='--', marker='o')
    ax[0].plot(data_XR['Xbar_UCL'], color='firebrick', linewidth=1)
    ax[0].plot(data_XR['Xbar_CL'], color='g', linewidth=1)
    ax[0].plot(data_XR['Xbar_LCL'], color='firebrick', linewidth=1)
    ax[0].set_ylabel('Sample Mean')
            # add the values of the control limits on the right side of the plot
    ax[0].text(len(data_XR)+.5, data_XR['Xbar_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XR['Xbar_UCL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XR)+.5, data_XR['Xbar_CL'].iloc[0], 'CL = {:.3f}'.format(data_XR['Xbar_CL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XR)+.5, data_XR['Xbar_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XR['Xbar_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
    ax[0].plot(data_XR['Xbar_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

    ax[1].plot(data_XR['sample_range'], color='mediumblue', linestyle='--', marker='o')
    ax[1].plot(data_XR['R_UCL'], color='firebrick', linewidth=1)
    ax[1].plot(data_XR['R_CL'], color='g', linewidth=1)
    ax[1].plot(data_XR['R_LCL'], color='firebrick', linewidth=1)
    ax[1].set_ylabel('Sample Range')
    ax[1].set_xlabel('Sample Number')
            # add the values of the control limits on the right side of the plot
    ax[1].text(len(data_XR)+.5, data_XR['R_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XR['R_UCL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(data_XR)+.5, data_XR['R_CL'].iloc[0], 'CL = {:.3f}'.format(data_XR['R_CL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(data_XR)+.5, data_XR['R_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XR['R_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
    ax[1].plot(data_XR['R_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            # set the x-axis limits
    ax[1].set_xlim(-1, len(data_XR))

    if subset_size!=None:
        ax[0].axvline(x=subset_size-.5, color='k', linestyle='--')
        ax[1].axvline(x=subset_size-.5, color='k', linestyle='--')

    plt.show()

    return data_XR

def xbarR_prob(mean_or_mu,std_or_sigma,n,alpha,sample_mean, sample_R,D_1_alpha2,Dalpha2,subset_size=None):
    K=stats.norm.ppf(1-alpha/2)
    data_XR = pd.DataFrame(columns=['Xbar_CL', 'Xbar_UCL', 'Xbar_LCL','R_UCL','R_LCL'])

    # Compute the CL, UCL and LCL for Xbar and S
    Xbar_CL = mean_or_mu
    Xbar_UCL = mean_or_mu + K*std_or_sigma/np.sqrt(n)
    Xbar_LCL = mean_or_mu - K*std_or_sigma/np.sqrt(n)

    R_UCL = D_1_alpha2 * std_or_sigma 
    R_LCL = Dalpha2 * std_or_sigma 

    if R_LCL < 0:
        R_LCL=0

    for i in range(len(sample_mean)):
        data_XR.loc[i] = [Xbar_CL, Xbar_UCL, Xbar_LCL, R_UCL, R_LCL]
    data_XR['sample_mean']=sample_mean
    data_XR['sample_range']=sample_R

    data_XR['Xbar_TEST1'] = np.where((data_XR['sample_mean'] > data_XR['Xbar_UCL']) | 
                    (data_XR['sample_mean'] < data_XR['Xbar_LCL']), data_XR['sample_mean'], np.nan)
    data_XR['R_TEST1'] = np.where((data_XR['sample_range'] > data_XR['R_UCL']) | 
                    (data_XR['sample_range'] < data_XR['R_LCL']), data_XR['sample_range'], np.nan)

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle(('Xbar-R charts'))
    ax[0].plot(data_XR['sample_mean'], color='mediumblue', linestyle='--', marker='o')
    ax[0].plot(data_XR['Xbar_UCL'], color='firebrick', linewidth=1)
    ax[0].plot(data_XR['Xbar_CL'], color='g', linewidth=1)
    ax[0].plot(data_XR['Xbar_LCL'], color='firebrick', linewidth=1)
    ax[0].set_ylabel('Sample Mean')
            # add the values of the control limits on the right side of the plot
    ax[0].text(len(data_XR)+.5, data_XR['Xbar_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XR['Xbar_UCL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XR)+.5, data_XR['Xbar_CL'].iloc[0], 'CL = {:.3f}'.format(data_XR['Xbar_CL'].iloc[0]), verticalalignment='center')
    ax[0].text(len(data_XR)+.5, data_XR['Xbar_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XR['Xbar_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
    ax[0].plot(data_XR['Xbar_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

    ax[1].plot(data_XR['sample_range'], color='mediumblue', linestyle='--', marker='o')
    ax[1].plot(data_XR['R_UCL'], color='firebrick', linewidth=1)
    ax[1].plot(data_XR['R_LCL'], color='firebrick', linewidth=1)
    ax[1].set_ylabel('Sample Range')
    ax[1].set_xlabel('Sample Number')
            # add the values of the control limits on the right side of the plot
    ax[1].text(len(data_XR)+.5, data_XR['R_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XR['R_UCL'].iloc[0]), verticalalignment='center')
    ax[1].text(len(data_XR)+.5, data_XR['R_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XR['R_LCL'].iloc[0]), verticalalignment='center')
            # highlight the points that violate the alarm rules
    ax[1].plot(data_XR['R_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
            # set the x-axis limits
    ax[1].set_xlim(-1, len(data_XR))

    if subset_size!=None:
        ax[0].axvline(x=subset_size-.5, color='k', linestyle='--')
        ax[1].axvline(x=subset_size-.5, color='k', linestyle='--')

    plt.show()
    return data_XR 

def IMR_my(sample_means,n,k,mean_or_mu,sigma_v,mean_to_est,sigma_v_to_est,show_MR='no',subset_size=None):

    df = pd.DataFrame({'I': sample_means})

    if show_MR!='no': #yes
        df['MR']=sample_means.diff().abs()
 
    d2 = qda.constants.getd2(2)
    d3= qda.constants.getd2(2)
    D4 = qda.constants.getD4(2)

        
    if mean_to_est=='yes' and sigma_v_to_est=='yes':
        df['I_UCL'] = mean_or_mu + (k*sigma_v)
        df['I_CL'] = mean_or_mu
        df['I_LCL'] = mean_or_mu - (k*sigma_v)
        if show_MR!='no':
            df['MR_UCL'] = D4 * df['MR'].iloc[:subset_size].mean()
            df['MR_CL'] = df['MR'].iloc[:subset_size].mean()
            df['MR_LCL'] = 0

    elif mean_to_est=='yes' and sigma_v_to_est!='yes':
        df['I_UCL'] = mean_or_mu + (k*sigma_v)
        df['I_CL'] = mean_or_mu
        df['I_LCL'] = mean_or_mu - (k*sigma_v)
        if show_MR!='no':
            df['MR_UCL'] = d2*sigma_v*np.sqrt(n) + k*d3*sigma_v*np.sqrt(n)
            df['MR_CL'] = d2*sigma_v*np.sqrt(n)
            df['MR_LCL'] = np.max([0,d2*sigma_v*np.sqrt(n) - k*d3*sigma_v*np.sqrt(n)])

    elif mean_to_est!='yes' and sigma_v_to_est!='yes':
        df['I_UCL'] = mean_or_mu + (k*sigma_v)
        df['I_CL'] = mean_or_mu
        df['I_LCL'] = mean_or_mu - (k*sigma_v)
        if show_MR!='no':
            df['MR_UCL'] = d2*sigma_v*np.sqrt(n)+ k*d3*sigma_v*np.sqrt(n)
            df['MR_CL'] = d2*sigma_v*np.sqrt(n)
            df['MR_LCL'] = np.max([0,d2*sigma_v*np.sqrt(n) - k*d3*sigma_v*np.sqrt(n)])

    elif mean_to_est!='yes' and sigma_v_to_est=='yes':
        df['I_UCL'] = mean_or_mu + (k*sigma_v)
        df['I_CL'] = mean_or_mu
        df['I_LCL'] = mean_or_mu - (k*sigma_v)
        if show_MR!='no':
            df['MR_UCL'] = d2*sigma_v*np.sqrt(n)+ k*d3*sigma_v*np.sqrt(n)
            df['MR_CL'] = d2*sigma_v*np.sqrt(n)
            df['MR_LCL'] = np.max([0,d2*sigma_v*np.sqrt(n) - k*d3*sigma_v*np.sqrt(n)])

    # Define columns for possible violations of the control limits
    df['I_TEST1'] = np.where((df['I'] > df['I_UCL']) | 
                    (df['I'] < df['I_LCL']), df['I'], np.nan)
    if show_MR!='no':
        df['MR_TEST1'] = np.where((df['MR'] > df['MR_UCL']) | 
                        (df['MR'] < df['MR_LCL']), df['MR'], np.nan)

    # Print the first 5 rows of the new dataframe
    df.head()

    # Plot the I chart
    plt.title('I chart')
    plt.plot(df['I'], color='b', linestyle='--', marker='o')
    plt.plot(df['I'], color='b', linestyle='--', marker='o')
    plt.plot(df['I_UCL'], color='r')
    plt.plot(df['I_CL'], color='g')
    plt.plot(df['I_LCL'], color='r')
    plt.ylabel('Individual Value')
    plt.xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    plt.text(len(df)+.5, df['I_UCL'].iloc[0], 'UCL = {:.3f}'.format(df['I_UCL'].iloc[0]), verticalalignment='center')
    plt.text(len(df)+.5, df['I_CL'].iloc[0], 'CL = {:.3f}'.format(df['I_CL'].iloc[0]), verticalalignment='center')
    plt.text(len(df)+.5, df['I_LCL'].iloc[0], 'LCL = {:.3f}'.format(df['I_LCL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(df['I_TEST1'], linestyle='none', marker='s', color='r', markersize=10)

    if subset_size!=None:
        plt.axvline(x=subset_size-.5, color='k', linestyle='--')
    plt.show()


    if show_MR!='no':
        plt.title('MR chart')
        plt.plot(df['MR'], color='b', linestyle='--', marker='o')
        plt.plot(df['MR_UCL'], color='r')
        plt.plot(df['MR_CL'], color='g')
        plt.plot(df['MR_LCL'], color='r')
        plt.ylabel('Moving Range')
        plt.xlabel('Sample number')
        # add the values of the control limits on the right side of the plot
        plt.text(len(df)+.5, df['MR_UCL'].iloc[0], 'UCL = {:.3f}'.format(df['MR_UCL'].iloc[0]), verticalalignment='center')
        plt.text(len(df)+.5, df['MR_CL'].iloc[0], 'CL = {:.3f}'.format(df['MR_CL'].iloc[0]), verticalalignment='center')
        plt.text(len(df)+.5, df['MR_LCL'].iloc[0], 'LCL = {:.3f}'.format(df['MR_LCL'].iloc[0]), verticalalignment='center')
        # highlight the points that violate the alarm rules
        plt.plot(df['MR_TEST1'], linestyle='none', marker='s', color='r', markersize=10)
        plt.show()
    return df

def reconstruct_from_pca_obj(data,pca,scores_df,k,from_standardize='yes'):
    mean = data.mean()
    std = data.std()

    # Compute the reconstructed data_std using the first k principal components
    #here clearly we're reconstructing all the dataset (not only the j-th observation):

    reconstructed_data_start = scores_df.iloc[0:,0:k].dot(pca.components_[0:k, :])

    # Now use the mean and standard deviation to compute the reconstructed data
    if from_standardize=='yes':
        reconstructed_data = reconstructed_data_start.dot(np.diag(std)) + np.asarray(mean)
    else:
        reconstructed_data = reconstructed_data_start.dot + np.asarray(mean)

    # Compare the original data with the reconstructed data
    print("Original data\n", data.head())
    print("\nReconstructed data\n", reconstructed_data.head())
    return reconstructed_data

def reconstruct_from_pca_df(data,eigenvectors,scores_df,k,mean,SIGMA,from_standardize='yes'):

    reconstructed_data_start = scores_df.iloc[0:,0:k].dot(eigenvectors[:,0:k].transpose())

    # Now use the mean and standard deviation to compute the reconstructed data
    if from_standardize=='yes':
        reconstructed_data = reconstructed_data_start.dot(np.diag(SIGMA)) + np.asarray(mean)
    else:
        reconstructed_data = reconstructed_data_start + np.asarray(mean)

    # Compare the original data with the reconstructed data
    print("Original data\n", data.head())
    print("\nReconstructed data\n", reconstructed_data.head())
    return reconstructed_data

def dotplot (data):
    for i in range(1,len(data.columns)+1):
        plt.plot(data.iloc[:,i-1], linestyle='none', marker='o', label = data.columns[i-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def OCcurveXbarChart(data,delta,k=3):
    n=len(data.columns)

    beta = stats.norm.cdf(k - delta*np.sqrt(n)) - stats.norm.cdf(-k - delta*np.sqrt(n))
    # Plot the beta values
    plt.plot(delta, beta)
    plt.xlabel('Delta')
    plt.ylabel('Beta')
    plt.title('Operating characteristic curve')
    plt.show()

    ARL = 1/(1-beta)

    # Plot the ARL values as a function of delta
    plt.plot(delta, ARL)
    plt.xlabel('Delta')
    plt.ylabel('ARL')
    plt.title('Average run length')
    plt.show()

def boxcox_and_unstack(data,approx_to_0_if_nec='False'):
    [data_BC, lmbda] = stats.boxcox(data.stack())
    print('il Lambda migliore è= %.3f' % lmbda)
    if (lmbda < 0.10 and lmbda>-0.10):
        if(approx_to_0_if_nec!='False'):
            lmbda=0
            data_BC = stats.boxcox(data.stack(), lmbda=0) 
            print('ma il lambda usato è 0')
   
    # Plot a histogram of the transformed data
    plt.hist(data_BC)
    plt.show()

    _, p_value_SW = stats.shapiro(data_BC)
    print('p-value of the Shapiro-Wilk test: %.5f' % p_value_SW)

    data_BC_unstack = data_BC.reshape(data.shape) #l'input è lo shape dei dati originali
    # and convert it to a DataFrame
    data_BC_unstack = pd.DataFrame(data_BC_unstack, columns = data.columns) #conversione
   
    print('')
    print("head di data_BC_unstack")
    print(data_BC_unstack.head())
    return data_BC_unstack,lmbda

def extractOOCindx(CC_result,nomecolonna_test):
    OOC_idx = np.where(CC_result[nomecolonna_test].notnull())[0] #[0] serve solo per prendersi unicamente il valore
    #e non l'oggetto in formato tupla 

    # Print the index of the OOC points
    print('The index of the OOC point is: {}'.format(OOC_idx))
    return OOC_idx

def IMRcc_half_normal(data,nome_colonna_dati,alpha=0.0027,subset__size=None):
    K_alpha = stats.norm.ppf(1-alpha/2)
    if subset__size == None:
        data_IMR = qda.ControlCharts.IMR(data, nome_colonna_dati, K = K_alpha)
    else:
        data_IMR = qda.ControlCharts.IMR(data, nome_colonna_dati, K = K_alpha, subset_size=subset__size)
    D_UCL = np.sqrt(2) * stats.norm.ppf(1-alpha/4)
    D_LCL = np.sqrt(2) * stats.norm.ppf(1 - (1/2 - alpha/4))

    MR_UCL = D_UCL * data_IMR['MR'].iloc[:subset__size].mean()/qda.constants.getd2(2)
    MR_LCL = D_LCL * data_IMR['MR'].iloc[:subset__size].mean()/qda.constants.getd2(2)
    data_IMR['MR_TEST1'] = np.where((data_IMR['MR'] > MR_UCL) | 
                                (data_IMR['MR'] < MR_LCL), data_IMR['MR'], np.nan)

    #plotta tutti gli MR
    plt.plot(data_IMR['MR'], 'o-')
    plt.axhline(MR_UCL, color = 'r')
    plt.axhline(MR_LCL, color = 'r')
    #plotta gli MR ooc marcandoli (in questo caso c'è un MR=0, quindi marcato come OOC)
    plt.plot(data_IMR['MR_TEST1'], linestyle='none', marker='s', color='r', markersize=10)
    plt.title("MR chart with using half-normal")
    if subset__size!=None:
        plt.axvline(x=(subset__size-1)-0.5, color='k', linestyle='--')
    plt.show()
    return data_IMR 

def IMRccboxcox(data, nome_colonna_dati, alpha=0.0027,subset__size='None'):
    K_alpha = stats.norm.ppf(1-alpha/2)
    if subset__size!=None:
        data_IMR = qda.ControlCharts.IMR(data, nome_colonna_dati, K = K_alpha,subset_size=subset__size)
    else:
        data_IMR = qda.ControlCharts.IMR(data, nome_colonna_dati, K = K_alpha)

    MR = pd.DataFrame(data_IMR['MR'])
    MR['MR'] = MR['MR'].transform(lambda x: ((x**0.4)))
    data_MR_transformed_dropna = MR.dropna()
    _, p_value_SW = stats.shapiro(data_MR_transformed_dropna['MR'])
    print('p-value of the Shapiro-Wilk test: %.3f' % p_value_SW)
    MR = MR.rename(columns = {'MR': 'MR_transformed'})
    data_MR_transformed = qda.ControlCharts.IMR(MR, 'MR_transformed', K = K_alpha, plotit = False, subset__size=subset__size)

    # Plot the I chart with the transformed data
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(('I charts of MR_transformed'))
    ax.plot(data_MR_transformed['MR_transformed'], color='mediumblue', linestyle='--', marker='o')
    ax.plot(data_MR_transformed['I_UCL'], color='firebrick', linewidth=1)
    ax.plot(data_MR_transformed['I_CL'], color='g', linewidth=1)
    ax.plot(data_MR_transformed['I_LCL'], color='firebrick', linewidth=1)
    ax.set_ylabel('Individual Value')
    ax.set_xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    ax.text(len(data_MR_transformed)+3, data_MR_transformed['I_UCL'].iloc[0], 'UCL = {:.2f}'.format(data_MR_transformed['I_UCL'].iloc[0]), verticalalignment='center')
    ax.text(len(data_MR_transformed)+3, data_MR_transformed['I_CL'].iloc[0], 'CL = {:.2f}'.format(data_MR_transformed['I_CL'].iloc[0]), verticalalignment='center')
    ax.text(len(data_MR_transformed)+3, data_MR_transformed['I_LCL'].iloc[0], 'LCL = {:.2f}'.format(data_MR_transformed['I_LCL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    ax.plot(data_MR_transformed['I_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)

    if subset__size!=None:
        ax.axvline(x=subset__size-0.5, color='k', linestyle='--')
    
    plt.show()

    return data_MR_transformed

def IMR_TrendCC(data_IMR, residual,fittedvalues, datapositive='no',alpha=0.0027):
    k=stats.norm.ppf(1-alpha/2)
    d2 = qda.constants.getd2(2)
    #df_res = pd.DataFrame({'I': model.resid})
    df_res = pd.DataFrame({'I': residual})
    df_res['MR'] = df_res['I'].diff().abs() 
    #data_IMR['I_CL'] = model.fittedvalues
    data_IMR['I_CL'] = fittedvalues

    # Replace the I_UCL and I_LCL columns with the upper and 
    # lower control limits computed from the formula
    data_IMR['I_UCL'] = data_IMR['I_CL'] + k * df_res['MR'].mean() / d2
    data_IMR['I_LCL'] = data_IMR['I_CL'] - k * df_res['MR'].mean() / d2

    # Also update the TEST1 column
    data_IMR['I_TEST1'] = np.where((data_IMR.iloc[:,0] > data_IMR['I_UCL']) | (data_IMR.iloc[:,0] < data_IMR['I_LCL']), data_IMR.iloc[:,0], np.nan)
    if (datapositive!='no'):
        data_IMR['I_LCL'] = np.where((data_IMR['I_LCL'] < 0), 0, data_IMR['I_LCL'])
    plt.title('I chart')
    plt.plot(data_IMR.iloc[:,0], color='b', linestyle='--', marker='o')
    plt.plot(data_IMR.iloc[:,0], color='b', linestyle='--', marker='o')
    plt.plot(data_IMR['I_UCL'], color='r')
    plt.plot(data_IMR['I_CL'], color='g')
    plt.plot(data_IMR['I_LCL'], color='r')
    plt.ylabel('Individual Value')
    plt.xlabel('Sample number')
    # highlight the points that violate the alarm rules
    plt.plot(data_IMR['I_TEST1'], linestyle='none', marker='s', 
            color='r', markersize=10)

    plt.show()
    return data_IMR,df_res

def IMR_SCC (df_res, alpha=0.0027):
    k=stats.norm.ppf(1-alpha/2)
    d2 = qda.constants.getd2(2)
    D4 = qda.constants.getD4(2)
    df_res['I_UCL'] = df_res['I'].mean() + (k*df_res['MR'].mean()/d2)
    df_res['I_CL'] = df_res['I'].mean()
    df_res['I_LCL'] = df_res['I'].mean() - (k*df_res['MR'].mean()/d2)
    df_res['MR_UCL'] = D4 * df_res['MR'].mean()
    df_res['MR_CL'] = df_res['MR'].mean()
    df_res['MR_LCL'] = 0

    # Define columns for the Western Electric alarm rules
    df_res['I_TEST1'] = np.where((df_res['I'] > df_res['I_UCL']) | 
                (df_res['I'] < df_res['I_LCL']), df_res['I'], np.nan)
    df_res['MR_TEST1'] = np.where((df_res['MR'] > df_res['MR_UCL']) | 
                (df_res['MR'] < df_res['MR_LCL']), df_res['MR'], np.nan)
    # Plot the I chart
    plt.title('I chart')
    plt.plot(df_res['I'], color='b', linestyle='--', marker='o')
    plt.plot(df_res['I'], color='b', linestyle='--', marker='o')
    plt.plot(df_res['I_UCL'], color='r')
    plt.plot(df_res['I_CL'], color='g')
    plt.plot(df_res['I_LCL'], color='r')
    plt.ylabel('Individual Value')
    plt.xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    # print the first value of the column I_UCL
    plt.text(len(df_res)+.5, df_res['I_UCL'].iloc[0], 
            'UCL = {:.2f}'.format(df_res['I_UCL'].iloc[0]), 
            verticalalignment='center')
    plt.text(len(df_res)+.5, df_res['I_CL'].iloc[0], 
            'CL = {:.2f}'.format(df_res['I_CL'].iloc[0]), 
            verticalalignment='center')
    plt.text(len(df_res)+.5, df_res['I_LCL'].iloc[0], 
            'LCL = {:.2f}'.format(df_res['I_LCL'].iloc[0]), 
            verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(df_res['I_TEST1'], linestyle='none', marker='s', 
            color='r', markersize=10)

    plt.show()

    plt.title('MR chart')
    plt.plot(df_res['MR'], color='b', linestyle='--', marker='o')
    plt.plot(df_res['MR_UCL'], color='r')
    plt.plot(df_res['MR_CL'], color='g')
    plt.plot(df_res['MR_LCL'], color='r')
    plt.ylabel('Moving Range')
    plt.xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    plt.text(len(df_res)+.5, df_res['MR_UCL'].iloc[0], \
            'UCL = {:.2f}'.format(df_res['MR_UCL'].iloc[0]), 
            verticalalignment='center')
    plt.text(len(df_res)+.5, df_res['MR_CL'].iloc[0], 
            'CL = {:.2f}'.format(df_res['MR_CL'].iloc[0]), 
            verticalalignment='center')
    plt.text(len(df_res)+.5, df_res['MR_LCL'].iloc[0], 
            'LCL = {:.2f}'.format(df_res['MR_LCL'].iloc[0]), 
            verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(df_res['MR_TEST1'], linestyle='none', marker='s', 
            color='r', markersize=10)

    plt.show()
    return df_res

def checkOOC(dati,datiLCL,datiUCL):
    result_IC = (dati >= datiLCL) & (dati <= datiUCL)
    return result_IC


def vettorializza(data):
    # Transpose the dataset and stack the columns
    data_stack = data.transpose().melt()

    # Remove unnecessary columns
    data_stack = data_stack.drop('variable', axis=1)

    print(data_stack.head())
    return data_stack

def IonMRresBoxCox(df_SCC,use04='no',alpha=0.0027,subset__size='False'):

    k=stats.norm.ppf(1-alpha/2)
    [data_norm, lambdozzo] = stats.boxcox(df_SCC['MR'].iloc[1:])
    print('Lambda = %.3f' % lambdozzo)

    if (use04!='no'):
        data_norm =df_SCC['MR'].iloc[1:].transform(lambda x: ((x**0.4))) 
        print('Ma il lambda usato è 0.4 come da scelta, per shapiro:')

    _, p_value_SW = stats.shapiro(data_norm)
    print('p-value of the Shapiro-Wilk test: %.3f' % p_value_SW)
    if (p_value_SW>0.05):
        print('box cox succeeded :)')

    df_MR_boxcox=pd.DataFrame({'MR_boxcox': data_norm})
    if subset__size!='False':
        df_MR_boxcox = qda.ControlCharts.IMR(df_MR_boxcox, 'MR_boxcox', plotit=False,K=k,subset_size=subset__size-1)
        #il -1 al subset size è messo per compensare il fatto che ho gli MR sono 1 in meno rispetto ai dati
        #poichè data_norm è ciò che finisce nella funzione in sostanza e quello ha 1 campione in meno rispetto
        #alla vera subset_size, ossia il numero di sample in phase1
    else:
         df_MR_boxcox = qda.ControlCharts.IMR(df_MR_boxcox, 'MR_boxcox', plotit=False,K=k)

    #plotit=False solo perchè non vogliamo entrambi e plot e ci mettiamo a costruire l'I chart sull'MR normalizzato dei residui

    print('')
    print('ricorda che ad eg. LCL di questo individual chart sugli MR trasformati dei residui che chiamo y(individuals), è trovato come ybar-3*(MRbar_y/d2(2)), dove MRbar_y è MR medio calcolato sugli y')
    # Plot the I chart
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(('I chart of MR_boxcox'))
    ax.plot(df_MR_boxcox['MR_boxcox'], color='mediumblue', linestyle='--', marker='o')
    ax.plot(df_MR_boxcox['I_UCL'], color='firebrick', linewidth=1)
    ax.plot(df_MR_boxcox['I_CL'], color='g', linewidth=1)
    ax.plot(df_MR_boxcox['I_LCL'], color='firebrick', linewidth=1)
    ax.set_ylabel('Individual Value')
    ax.set_xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    ax.text(len(df_MR_boxcox)+.5, df_MR_boxcox['I_UCL'].iloc[0], 'UCL = {:.2f}'.format(df_MR_boxcox['I_UCL'].iloc[0]), verticalalignment='center')
    ax.text(len(df_MR_boxcox)+.5, df_MR_boxcox['I_CL'].iloc[0], 'CL = {:.2f}'.format(df_MR_boxcox['I_CL'].iloc[0]), verticalalignment='center')
    ax.text(len(df_MR_boxcox)+.5, df_MR_boxcox['I_LCL'].iloc[0], 'LCL = {:.2f}'.format(df_MR_boxcox['I_LCL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    ax.plot(df_MR_boxcox['I_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
    
    if subset__size !='False':
        ax.axvline(x=(subset__size-1)-.5, color='k', linestyle='--')

    plt.show()

    return df_MR_boxcox

def Rcc(data_XR):
    plt.title('R chart')
    plt.plot(data_XR['sample_range'], color='b', linestyle='--', marker='o')
    plt.plot(data_XR['R_UCL'], color='r')
    plt.plot(data_XR['R_CL'], color='g')
    plt.plot(data_XR['R_LCL'], color='r')
    plt.ylabel('Sample range')
    plt.xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    plt.text(len(data_XR)+.5, data_XR['R_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_XR['R_UCL'].iloc[0]), verticalalignment='center')
    plt.text(len(data_XR)+.5, data_XR['R_CL'].iloc[0], 'CL = {:.3f}'.format(data_XR['R_CL'].iloc[0]), verticalalignment='center')
    plt.text(len(data_XR)+.5, data_XR['R_LCL'].iloc[0], 'LCL = {:.3f}'.format(data_XR['R_LCL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(data_XR['R_TEST1'], linestyle='none', marker='s', color='r', markersize=10)
    plt.show()
   

def HotCC_phase1(p, m, n, alpha, sample_mean, Xbarbar, S, quanti_sample_levati=0):
    S_inv = np.linalg.inv(S)
    # m deve corrispondere sempre al numero di sample che si usano escludendo quindi eventuali NaN (sample tolti)
    # cosi calcolo UCL senza commettere errori:
    UCL = (p * (m-1) * (n-1)) / (m * (n-1) - (p-1)) * stats.f.ppf(1-alpha, p, m*n - m + 1 - p)
    T2 = []  # empty list

    # Calcolo dei valori T2 per ciascun campione
    for i in range(m + quanti_sample_levati):
        T2.append(n * (sample_mean.iloc[i] - Xbarbar).transpose().dot(S_inv).dot(sample_mean.iloc[i] - Xbarbar))

    # Plot the Hotelling T2 statistic
    plt.plot(T2, 'o-')
    plt.plot([0, m], [UCL, UCL], 'r-', label=f'UCL = {UCL:.2f}')  # plot UCL line
    plt.plot([0, m], [np.median(T2), np.median(T2)], 'g-', label=f'Median = {np.median(T2):.2f}')  # plot median line

    # Annotate UCL value on the plot
    plt.text(m - 1, UCL, f'UCL = {UCL:.2f}', color='red', ha='right', va='bottom')
    
    # Annotate median value on the plot
    plt.text(m - 1, np.median(T2), f'Median = {np.median(T2):.2f}', color='green', ha='right', va='bottom')

    # Adding labels and legend
    plt.xlabel('Sample')
    plt.ylabel('Hotelling T2')
    plt.legend()
    plt.show()

    return UCL, T2

def HotCC_phase2(p,m,n,alpha, sample_mean2,Xbarbar,S):
    c2 = (p*(n-1)*(m+1))/(m*(n-1)-(p-1))

    S_inv=np.linalg.inv(S)
    UCL = c2*stats.f.ppf(1-alpha, p, (m*(n-1)-(p-1)))
    T2 = [] #empty list
    for i in range(m):
        #append appende appunto i valori della T2 ad ogni iterazione
        T2.append(n * (sample_mean2.iloc[i]-Xbarbar).transpose().dot(S_inv).dot(sample_mean2.iloc[i]-Xbarbar))
    #fine ciclo

    # Plot the Hotelling T2 statistic
    plt.plot(T2, 'o-')
    plt.plot([0, m], [UCL, UCL], 'r-') #plotto riga rossa che rappresenta UCL
    plt.plot([0, m], [np.median(T2), np.median(T2)], 'g-')#plotto valore mediano della T2
    #spesso nei T2 control chart si plotta il valore mediano

    plt.xlabel('Sample')
    plt.ylabel('Hotelling T2')
    plt.show()
    return UCL,T2

def chi2cc(p,m,n,alpha,sample_means,mu,SIGMA):

    UCL = stats.chi2.ppf(1 - alpha, df = p)

    S_inv=np.linalg.inv(SIGMA)
    data_CC = sample_means.copy()
    data_CC['Chi2'] = np.nan

    for i in range(m):
        data_CC['Chi2'].iloc[i] = n * (sample_means.iloc[i] - mu).transpose().dot(S_inv).dot(sample_means.iloc[i] - mu)

    # Now we can add the UCL, CL and LCL to the dataframe
    data_CC['Chi2_UCL'] = UCL
    data_CC['Chi2_CL'] = data_CC['Chi2'].median()
    data_CC['Chi2_LCL'] = 0

    # Add one column to test if the sample is out of control
    data_CC['Chi2_TEST'] = np.where((data_CC['Chi2'] > data_CC['Chi2_UCL']), data_CC['Chi2'], np.nan)
    #se eccedi metti la chi2 se no metti nan

    plt.title('Chi2 control chart')
    plt.plot(data_CC['Chi2'], color='b', linestyle='--', marker='o')
    plt.plot(data_CC['Chi2_UCL'], color='r')
    plt.plot(data_CC['Chi2_CL'], color='g')
    plt.plot(data_CC['Chi2_LCL'], color='r')
    plt.ylabel('Chi2 statistic')
    plt.xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    plt.text(len(data_CC)+.5, data_CC['Chi2_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_CC['Chi2_UCL'].iloc[0]), verticalalignment='center')
    plt.text(len(data_CC)+.5, data_CC['Chi2_CL'].iloc[0], 'median = {:.3f}'.format(data_CC['Chi2_CL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(data_CC['Chi2_TEST'], linestyle='none', marker='s', color='r', markersize=10)
    plt.show()
    return UCL,data_CC

def T2cc_n_1(p,m,alpha,data,Xbar,S2):

    data_CC = data.copy()
    S2_inv = np.linalg.inv(S2)
    data_CC['T2'] = np.nan
    for i in range(m):
        data_CC['T2'].iloc[i] = (data.iloc[i] - Xbar).transpose().dot(S2_inv).dot(data.iloc[i] - Xbar)

    # Now we can add the UCL, CL and LCL to the dataframe
    cost=((m-1)**2)/m
    UCL=cost*stats.beta.ppf(1 - alpha, p/2, (m-p-1)/2)
    data_CC['T2_UCL'] = UCL
    data_CC['T2_CL'] = data_CC['T2'].median()
    data_CC['T2_LCL'] = 0

    # Add one column to test if the sample is out of control
    data_CC['T2_TEST'] = np.where((data_CC['T2'] > data_CC['T2_UCL']), data_CC['T2'], np.nan)
    plt.title('T2 control chart')
    plt.plot(data_CC['T2'], color='b', linestyle='--', marker='o')
    plt.plot(data_CC['T2_UCL'], color='r')
    plt.plot(data_CC['T2_CL'], color='g')
    plt.plot(data_CC['T2_LCL'], color='r')
    plt.ylabel('T2 statistic')
    plt.xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    plt.text(len(data_CC)+.5, data_CC['T2_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_CC['T2_UCL'].iloc[0]), verticalalignment='center')
    plt.text(len(data_CC)+.5, data_CC['T2_CL'].iloc[0], 'median = {:.3f}'.format(data_CC['T2_CL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(data_CC['T2_TEST'], linestyle='none', marker='s', color='r', markersize=10)
    plt.show()
    return UCL,data_CC

def T2cc_n_1_ph2(p,m,m_ph2,alpha,data_to_use_2,Xbar,S2):

    data_CC = data_to_use_2.copy()
    S2_inv = np.linalg.inv(S2)
    data_CC['T2'] = np.nan
    for i in range(m_ph2):
        data_CC['T2'].iloc[i] = (data_to_use_2.iloc[i] - Xbar).transpose().dot(S2_inv).dot(data_to_use_2.iloc[i] - Xbar)

    # Now we can add the UCL, CL and LCL to the dataframe
    cost=(p*(m+1)*(m-1))/((m**2)-m*p)
    UCL=cost*stats.f.ppf(1-alpha, p, m-p)
    data_CC['T2_UCL'] = UCL
    data_CC['T2_CL'] = data_CC['T2'].median()
    data_CC['T2_LCL'] = 0

    # Add one column to test if the sample is out of control
    data_CC['T2_TEST'] = np.where((data_CC['T2'] > data_CC['T2_UCL']), data_CC['T2'], np.nan)
    plt.title('T2 control chart')
    plt.plot(data_CC['T2'], color='b', linestyle='--', marker='o')
    plt.plot(data_CC['T2_UCL'], color='r')
    plt.plot(data_CC['T2_CL'], color='g')
    plt.plot(data_CC['T2_LCL'], color='r')
    plt.ylabel('T2 statistic')
    plt.xlabel('Sample number')
    # add the values of the control limits on the right side of the plot
    plt.text(len(data_CC)+.5, data_CC['T2_UCL'].iloc[0], 'UCL = {:.3f}'.format(data_CC['T2_UCL'].iloc[0]), verticalalignment='center')
    plt.text(len(data_CC)+.5, data_CC['T2_CL'].iloc[0], 'median = {:.3f}'.format(data_CC['T2_CL'].iloc[0]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(data_CC['T2_TEST'], linestyle='none', marker='s', color='r', markersize=10)
    plt.show()
    return UCL,data_CC

import numpy as np
import pandas as pd

def calculate_d(sample_mean, Xbarbar, S):
    m, p = sample_mean.shape  # Number of samples (m) and number of variables (p)
    n = 1  # Assuming that each sample mean represents an average of `n` observations
    
    S_inv = np.linalg.inv(S)
    T2_overall = []

    # Calculate the overall T^2 statistic for each sample
    for i in range(m):
        T2 = n * (sample_mean.iloc[i] - Xbarbar.values).transpose().dot(S_inv).dot(sample_mean.iloc[i] - Xbarbar.values)
        T2_overall.append(T2)
    
    dji_df = pd.DataFrame(index=sample_mean.index, columns=sample_mean.columns)
    
    # Iterate over each variable (column)
    for j in range(p):
        # Remove the j-th column
        remaining_vars = sample_mean.columns.drop(sample_mean.columns[j])
        
        # Compute the new mean vector and covariance matrix excluding the j-th variable
        Xbarbar_reduced = Xbarbar[remaining_vars]
        S_reduced = S.loc[remaining_vars, remaining_vars]
        S_inv_reduced = np.linalg.inv(S_reduced)
        
        # Calculate the T^2 statistic for each sample, excluding the j-th variable
        for i in range(m):
            sample_mean_reduced = sample_mean.iloc[i][remaining_vars]
            T2_reduced = n * (sample_mean_reduced - Xbarbar_reduced.values).transpose().dot(S_inv_reduced).dot(sample_mean_reduced - Xbarbar_reduced.values)
            
            # Calculate d_ji as the difference between the overall T^2 and the reduced T^2
            dji_df.iloc[i, j] = T2_overall[i] - T2_reduced
            
    return dji_df

# Example usage (assuming you have the necessary inputs):
# sample_mean = pd.DataFrame(...)  # m samples x p variables DataFrame
# Xbarbar = pd.Series(...)  # p-dimensional mean vector
# S = pd.DataFrame(...)  # p x p covariance matrix

# dji_df = calculate_dji(sample_mean, Xbarbar,


def cusum(data,col_means_name,mu_or_xbarbar,sigma_xbar,h=4,k=0.5):
   
    H = h*sigma_xbar #control limits
    K = k*sigma_xbar

    df_CUSUM = data.copy()
    df_CUSUM['Ci+'] = 0.0
    df_CUSUM['Ci-'] = 0.0

    for i in range(len(df_CUSUM)):
        if i == 0:
            df_CUSUM.loc[i, 'Ci+'] = max(0, df_CUSUM.loc[i, col_means_name] - (mu_or_xbarbar + K))
            df_CUSUM.loc[i, 'Ci-'] = max(0, (mu_or_xbarbar - K) - df_CUSUM.loc[i, col_means_name])
        else:
            df_CUSUM.loc[i, 'Ci+'] = max(0, df_CUSUM.loc[i, col_means_name] - (mu_or_xbarbar + K) + df_CUSUM.loc[i-1, 'Ci+'])
            df_CUSUM.loc[i, 'Ci-'] = max(0, (mu_or_xbarbar - K) - df_CUSUM.loc[i, col_means_name] + df_CUSUM.loc[i-1, 'Ci-'])

    df_CUSUM['Ci+_TEST1'] = np.where((df_CUSUM['Ci+'] > H) | (df_CUSUM['Ci+'] < -H), df_CUSUM['Ci+'], np.nan)
    df_CUSUM['Ci-_TEST1'] = np.where((df_CUSUM['Ci-'] > H) | (df_CUSUM['Ci-'] < -H), df_CUSUM['Ci-'], np.nan)

    plt.hlines(H, 0, len(df_CUSUM), color='firebrick', linewidth=1)
    plt.hlines(0, 0, len(df_CUSUM), color='g', linewidth=1)
    plt.hlines(-H, 0, len(df_CUSUM), color='firebrick', linewidth=1)
    # Plot the chart
    plt.title('CUSUM chart of %s (h=%.2f, k=%.2f)' % (col_means_name, h, k))
    plt.plot(df_CUSUM['Ci+'], color='b', linestyle='-', marker='o')
    plt.plot(-df_CUSUM['Ci-'], color='b', linestyle='-', marker='D')
    # add the values of the control limits on the right side of the plot
    plt.text(len(df_CUSUM)+.5, H, 'UCL = {:.3f}'.format(H), verticalalignment='center')
    plt.text(len(df_CUSUM)+.5, 0, 'CL = {:.3f}'.format(0), verticalalignment='center')
    plt.text(len(df_CUSUM)+.5, -H, 'LCL = {:.3f}'.format(-H), verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(df_CUSUM['Ci+_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
    plt.plot(-df_CUSUM['Ci-_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
    plt.xlim(-1, len(df_CUSUM))
    plt.show()

    return df_CUSUM

def EWMA(data,col_means_name,mu_or_xbarbar,sigma_xbar,lambda_):
    df_EWMA = data.copy()
    df_EWMA['a_t'] = lambda_/(2-lambda_) * (1 - (1-lambda_)**(2*np.arange(1, len(df_EWMA)+1)))
    for i in range(len(df_EWMA)):
        if i == 0: #la formula è leggermente diversa nell'inizializzazione
            df_EWMA.loc[i, 'z'] = lambda_*df_EWMA.loc[i, col_means_name] + (1-lambda_)*mu_or_xbarbar
        else:
            df_EWMA.loc[i, 'z'] = lambda_*df_EWMA.loc[i, col_means_name] + (1-lambda_)*df_EWMA.loc[i-1, 'z']
    df_EWMA['UCL'] = mu_or_xbarbar + 3*sigma_xbar*np.sqrt(df_EWMA['a_t'])
    df_EWMA['CL'] = mu_or_xbarbar
    df_EWMA['LCL'] = mu_or_xbarbar - 3*sigma_xbar*np.sqrt(df_EWMA['a_t'])

    df_EWMA['z_TEST1'] = np.where((df_EWMA['z'] > df_EWMA['UCL']) | (df_EWMA['z'] < df_EWMA['LCL']), df_EWMA['z'], np.nan)
    
    plt.plot(df_EWMA['UCL'], color='firebrick', linewidth=1)
    plt.plot(df_EWMA['CL'], color='g', linewidth=1)
    plt.plot(df_EWMA['LCL'], color='firebrick', linewidth=1)
    # Plot the chart
    plt.title('EWMA chart of %s (lambda=%.2f)' % (col_means_name, lambda_))
    plt.plot(df_EWMA['z'], color='b', linestyle='-', marker='o')
    # add the values of the control limits on the right side of the plot
    plt.text(len(df_EWMA)+.5, df_EWMA['UCL'].iloc[-1], 'UCL = {:.3f}'.format(df_EWMA['UCL'].iloc[-1]), verticalalignment='center')
    plt.text(len(df_EWMA)+.5, df_EWMA['CL'].iloc[-1], 'CL = {:.3f}'.format(df_EWMA['CL'].iloc[-1]), verticalalignment='center')
    plt.text(len(df_EWMA)+.5, df_EWMA['LCL'].iloc[-1], 'LCL = {:.3f}'.format(df_EWMA['LCL'].iloc[-1]), verticalalignment='center')
    # highlight the points that violate the alarm rules
    plt.plot(df_EWMA['z_TEST1'], linestyle='none', marker='s', color='firebrick', markersize=10)
    plt.xlim(-1, len(df_EWMA))
    plt.show()
    return df_EWMA