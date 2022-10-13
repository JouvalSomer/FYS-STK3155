import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import seaborn as sns


from func import*


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["ComputerModern"]})
plt.rcParams['figure.figsize'] = (8,6)

plt.rcParams.update({'font.size': 20})

""""""
def FF_Plot(Z, x, y, noise, save):
    plt.rc('axes', facecolor='none', edgecolor='none',
       axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=1)
    fig = plt.figure()
    # fig.patch.set_facecolor('whitesmoke')
    ax = fig.gca(projection='3d')

    ax.w_xaxis.set_pane_color((230/255, 230/255, 230/255, 1))
    ax.w_yaxis.set_pane_color((230/255, 230/255, 230/255, 1))
    ax.w_zaxis.set_pane_color((230/255, 230/255, 230/255, 1))

    surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


    cbaxes = fig.add_axes([0.12, 0.28, 0.03, 0.4])
    plt.colorbar(surf, cax = cbaxes, shrink=0.5, aspect=5)

    ax.set_title(r'\bf{Frank Function Surface Plot}', y=0.94)
    ax.set_xlabel(r'x values', labelpad=15)
    ax.set_ylabel(r'y values', labelpad=15)
    ax.set_zlabel(r'Frank Funcrion', labelpad=15)
    ax.tick_params(axis='both', which='major')
    ax.view_init(elev=12, azim=-37)
    plt.tight_layout()
    if save and noise:
        plt.savefig(f'plots/FF_Surface_Plot/FF_Surface_Plot_N.png', dpi=400)
        plt.clf()
    if save and not noise:
        plt.savefig(f'plots/FF_Surface_Plot/FF_Surface_Plot.png', dpi=400)
        plt.clf()

""""""
def FF_OLS_Aprox_plot(x, y, z_tilde, save):
    plt.rc('axes', facecolor='none', edgecolor='none',
       axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=1)
    fig = plt.figure()
    # fig.patch.set_facecolor('whitesmoke')
    ax = fig.gca(projection='3d')

    ax.w_xaxis.set_pane_color((230/255, 230/255, 230/255, 1))
    ax.w_yaxis.set_pane_color((230/255, 230/255, 230/255, 1))
    ax.w_zaxis.set_pane_color((230/255, 230/255, 230/255, 1))

    surf_tilde = ax.plot_surface(x, y, z_tilde, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    cbaxes = fig.add_axes([0.12, 0.3, 0.04, 0.4])
    plt.colorbar(surf_tilde, cax = cbaxes, shrink=0.5, aspect=5)

    ax.set_title(r'\bf{Frank Function Surface Plot - Regression model}', y=0.94)
    ax.set_xlabel(r'x values', labelpad=5)
    ax.set_ylabel(r'y values', labelpad=5)
    ax.set_zlabel(r'Frank Funcrion', labelpad=5)
    ax.tick_params(axis='both', which='major')
    ax.view_init(elev=12, azim=-37)
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/FF_OLS_Aprox_plot/FF_OLS_Aprox_plot.png', dpi=400)
        plt.clf()

""""""
def OLS_MSE_r2_plot(x, y, z, order, seed, mse_plot, r2_plot, scale, bootstrap, n_boostraps, CV, k, save):
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
       axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    poly_deg = np.arange(order)
    MSE_test = np.zeros(order)
    R2_test = np.zeros(order)

    if bootstrap or CV:
        mse_train, MSE_test, R2_train, R2_test, bias, variance = OLS_Reg(x, y, z, order, seed, scale, bootstrap, n_boostraps, CV, k)

    else:
        for i in range(order):
            MSE_train, MSE_test[i], R2_train, R2_test[i], beta, z_tilde_test, bias, variance = OLS_Reg(x, y, z, i, seed, scale, bootstrap, n_boostraps, CV, k)

    if mse_plot==True:
        index = np.argwhere(MSE_test == np.min(MSE_test))

        if index[0,0] == 0:
            index[0,0] += 1

        best_poly_deg_MSE = poly_deg[index[0,0]]
        order_updated_MSE = int(np.ceil(1.5*best_poly_deg_MSE))

        if CV:
            print(f'OLS - Lowest MSE = {np.min(MSE_test):.4f} and was achieved using CV is found at polynomial degree = {best_poly_deg_MSE}.')

        else:
            print(f'OLS - Lowest MSE = {np.min(MSE_test):.4f} and was achieved using no resampling is found at polynomial degree = {best_poly_deg_MSE}.')

        plt.plot(poly_deg[:order_updated_MSE], MSE_test[:order_updated_MSE], color='crimson', lw=2, label=r"Mean MSE for the test data") #zorder=0,
        plt.scatter(best_poly_deg_MSE, np.min(MSE_test), color='forestgreen', marker='x', zorder=100, s=150, label='Lowest MSE')
        plt.xlabel(r"Polynomial degree", labelpad=10)
        plt.ylabel(r"MSE", labelpad=10)
        if scale:
            plt.title(r"\bf{Mean Square Error - scaled}", pad=15)
        else:
            plt.title(r"\bf{Mean Square Error - not scaled}", pad=15)
        plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        plt.tight_layout()
        if save and CV:
            plt.savefig(f'plots/mse_plots/OLS_mean_mse_CV.png', dpi=400)
            plt.clf()
        if save and bootstrap:
            plt.savefig(f'plots/mse_plots/OLS_mean_mse_BS.png', dpi=400)
            plt.clf()
        elif save:
            plt.savefig(f'plots/mse_plots/OLS_mean_mse_nboot{n_boostraps}.png', dpi=400)
            plt.clf()

    if r2_plot==True:
        index = np.argwhere(R2_test == np.max(R2_test))
        best_poly_deg_R2 = poly_deg[index[0,0]]
        order_updated_MSE = int(np.ceil(1.5*best_poly_deg_R2))

        if CV:
            print(f'OLS - Highest R2 score = {np.max(R2_test):.4f} and was achieved using CV is found at polynomial degree = {best_poly_deg_MSE}.')

        else:
            print(f'OLS - Highest R2 score  = {np.max(R2_test):.4f} and was achieved using no resampling is found at polynomial degree = {best_poly_deg_MSE}.')

        plt.plot(poly_deg[:order_updated_MSE], R2_test[:order_updated_MSE], color='darkorange', lw=2, label=r"R2 score for the test data") #zorder=0,
        plt.scatter(best_poly_deg_R2, np.max(R2_test), color='blue', marker='x', zorder=100, s=150, label='Lowest MSE')
        plt.xlabel(r"Polynomial degree", labelpad=10)
        plt.ylabel(r"$R^2$", labelpad=10)
        if scale:
            plt.title(r"\bf{$R^2$ score - scaled}", pad=15)
        else:
            plt.title(r"\bf{$R^2$ score - not scaled}", pad=15)

        plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/R2_plots/OLS_mean_R2_nboot{n_boostraps}.png', dpi=400)
            plt.clf()

""""""
def bvto_OLS(x, y, z, order, seed,scale, bootstrap, n_boostraps, save):
    poly_deg = np.arange(order)
    MSE_test = np.zeros(order)
    Bias = np.zeros(order)
    Variance = np.zeros(order)

    if bootstrap:
            mse_train, MSE_test, R2_train, R2_test, Bias, Variance = OLS_Reg(x, y, z, order, seed, scale, bootstrap, n_boostraps)
    else:
        for i in range(order):
            MSE_train, MSE_test[i], R2_train, R2_test, beta, z_tilde_test, Bias[i], Variance[i] = OLS_Reg(x, y, z, i, seed, scale, bootstrap, n_boostraps, CV=False, k=10)

    index = np.argwhere(MSE_test == np.min(MSE_test))
    best_poly_deg_MSE = poly_deg[index[0,0]]
    order_updated_MSE = int(np.ceil(3*best_poly_deg_MSE))

    if order_updated_MSE > order:
        order_updated_MSE = order

    if bootstrap:
        print(f'OLS - Lowest MSE {np.min(MSE_test):.4f} using BS is found at polynomial degree = {best_poly_deg_MSE}.')

    else:
        print(f'OLS - Lowest MSE {np.min(MSE_test):.4f} using no resampling is found at polynomial degree = {best_poly_deg_MSE}.')

    plt.plot(poly_deg[:order_updated_MSE], MSE_test[:order_updated_MSE], color='crimson', lw=3, label=r"MSE")
    plt.plot(poly_deg[:order_updated_MSE], Bias[:order_updated_MSE], color='limegreen', lw=2, label=r"Bias")
    plt.plot(poly_deg[:order_updated_MSE], Variance[:order_updated_MSE], color='dodgerblue', lw=2, label=r"Variance")
    plt.scatter(best_poly_deg_MSE, np.min(MSE_test), color='black', marker='x', zorder=100, s=150, label='Lowest MSE')

    # plt.yscale('log')
    # plt.ylim(-0.0001, 0.04)
    plt.xlabel(r"Polynomial degree", labelpad=10)
    plt.ylabel(r"Error", labelpad=10)
    plt.title(r"\bf{Bias-Variance Trade-off}", pad=15)
    plt.legend(framealpha=0.9, facecolor=(255/255, 255/255, 255/255, 1))
    #plt.tight_layout()
    if save:
        plt.savefig(f'plots/bv_plots/OLS_BVTO_nboot{n_boostraps}.png', dpi=400)
        plt.clf()

""""""
def bvto_RIDGE_LASSO(x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps,CV, k, method, LASSOiter, LASSOtol, save):
    poly_deg = np.arange(order)
    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
    bias = np.empty((order, nlambdas))
    variance = np.empty((order, nlambdas))
    MSE_degree_lambda = np.empty((order, nlambdas))

    if method == 'RIDGE':
        MSE_degree_lambda, bias, variance = Ridge_Reg(x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps, CV, k)

    if method == 'LASSO':
        MSE_degree_lambda, bias, variance = Lasso_Reg(x, y, z, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps, CV, k, LASSOiter, LASSOtol)

    index = np.argwhere(MSE_degree_lambda == np.min(MSE_degree_lambda))
    best_poly_deg_MSE = poly_deg[index[0,0]]
    best_lambda_MSE_index = index[0,1]
    order_updated_MSE = int(np.ceil(3*best_poly_deg_MSE))

    if order_updated_MSE > order:
        order_updated_MSE = order

    print(f'{method} - Lowest MSE {np.min(MSE_degree_lambda):.4f} using BS is found at polynomial degree = {best_poly_deg_MSE}, and lambda = {lambdas[best_lambda_MSE_index]}.')

    plt.plot(poly_deg[:order_updated_MSE], MSE_degree_lambda[:order_updated_MSE, best_lambda_MSE_index], color='crimson', lw=3, label=r"MSE")
    plt.plot(poly_deg[:order_updated_MSE], bias[:order_updated_MSE, best_lambda_MSE_index], color='limegreen', lw=2, label=r"Bias")
    plt.plot(poly_deg[:order_updated_MSE], variance[:order_updated_MSE, best_lambda_MSE_index], color='dodgerblue', lw=2, label=r"Variance")
    plt.scatter(best_poly_deg_MSE, np.min(MSE_degree_lambda), color='black', marker='x', zorder=100, s=150, label='Lowest MSE')

    # plt.yscale('log')
    # plt.ylim(-0.0001, 0.04)
    plt.xlabel(r"Polynomial degree",labelpad=10)
    plt.ylabel(r"Error",labelpad=10)
    plt.title(r"\bf{Bias-Variance Trade-off}",pad=15)
    plt.legend(framealpha=0.9, facecolor=(255/255, 255/255, 255/255, 1))
    #plt.tight_layout()
    if save:
        plt.savefig(f'plots/bv_plots/{method}_BVTO_nboot{n_boostraps}.png', dpi=400)
        plt.clf()

""""""
def beta_plot(x_, y_, z_, order, seed, save, scale):
    plt.rcParams['figure.figsize'] = (12,9)
    plt.rcParams.update({'font.size': 26})
    plt.rc('axes', facecolor='whitesmoke', edgecolor='none',
       axisbelow=True, grid=True)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('lines', linewidth=2)

    poly_deg = np.arange(order)

    B0 = np.zeros(order)
    B1 = np.zeros(order)
    B2 = np.zeros(order)

    bootstrap = False
    n = 200
    for i in poly_deg[1:]:
        beta = OLS_Reg(x_, y_, z_, i, seed, scale, bootstrap, n)[4]
        B0[i] = beta[0]
        B1[i] = beta[1]
        B2[i] = beta[2]

    if save and not scale:
        plt.plot(poly_deg, B0, color = 'limegreen', lw=1.5, label=r'$\beta_0$')
        plt.plot(poly_deg, B1, color = 'dodgerblue', lw=1.5, label=r'$\beta_1$')
        plt.plot(poly_deg, B2, color = 'crimson', lw=1.5, label=r'$\beta_2$')
        plt.xlabel("Polynomial degree", labelpad=10)
        plt.ylabel(r'$\beta$ values', labelpad=10)
        plt.title(r"\bf{Beta Values Against Order of Polynomia}", pad=15)
        plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        plt.tight_layout()
        plt.savefig(f'plots/beta_plot/beta_plot.png', dpi=400)
        plt.clf()

    if save and scale:
        # plt.axvline(x = 18, color = 'dimgray', ls='--', alpha=0.5, label = 'Best poly. deg.')
        plt.plot(poly_deg, B0, color = 'limegreen', lw=1.5, label=r'$\beta_0$ scaled')
        plt.plot(poly_deg, B1, color = 'dodgerblue', lw=1.5, label=r'$\beta_1$ scaled')
        plt.plot(poly_deg, B2, color = 'crimson', lw=1.5, label=r'$\beta_2$ scaled')
        plt.xlabel("Polynomial degree", labelpad=10)
        plt.ylabel(r'Scaled $\beta$ values', labelpad=10)
        plt.title(r"\bf{Scaled Beta Values Against Order of Polynomia}", pad=15)
        plt.legend(framealpha=0.9, facecolor=(1, 1, 1, 1))
        plt.tight_layout()
        plt.savefig(f'plots/beta_plot/beta_plot_scaled_with_vline.png', dpi=400)
        plt.clf()


""" Heatmap for RIDGE and LASSO """
def MSE_heatmap_RIDGE_LASSO(method, x_, y_, z_, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps,  CV, k, LASSOiter, LASSOtol,save, data):
    fig, ax = plt.subplots(figsize=(16,12))
    plt.rcParams.update({'font.size': 26})
    plt.rcParams['axes.titlepad'] = 40 # Space between the title and graph

    lambdas = np.logspace(lambda_min, lambda_max, nlambdas)
    poly_deg = np.arange(order)

    if method == 'RIDGE':
        MSE_degree_lambda = Ridge_Reg(x_, y_, z_, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps, CV, k)[0]

    if method == 'LASSO':
        MSE_degree_lambda = Lasso_Reg(x_, y_, z_, order, seed, scale, nlambdas, lambda_min, lambda_max, bootstrap, n_boostraps, CV, k, LASSOiter, LASSOtol)[0]

    index = np.argwhere(MSE_degree_lambda == np.min(MSE_degree_lambda))

    if index[0,0] == 0:
        index[0,0] += 1

    if index[0,1] == 0:
        index[0,1] += 1

    best_poly_deg_MSE = poly_deg[index[0,0]]
    best_lambda_MSE_index = index[0,1]

    if CV:
        print(f'{method} - Lowest MSE {np.min(MSE_degree_lambda):.4f} using CV is found at polynomial degree = {best_poly_deg_MSE}, and lambda = {lambdas[best_lambda_MSE_index]}.')

    if bootstrap:
        print(f'{method} - Lowest MSE {np.min(MSE_degree_lambda):.4f} using BS is found at polynomial degree = {best_poly_deg_MSE}, and lambda = {lambdas[best_lambda_MSE_index]}.')

    max_order = int(np.ceil(1.5*best_poly_deg_MSE))
    min_order = int(np.floor(0.5*best_poly_deg_MSE))

    if (max_order - min_order) <= 2:
        max_order += 1

    max_lambda = int(np.ceil(1.5*best_lambda_MSE_index))
    min_lambda = int(np.floor(0.5*best_lambda_MSE_index))

    if (max_lambda - min_lambda) <= 2:
        max_lambda += 1

    if max_order > order:
        max_order = order

    if max_lambda > len(lambdas):
        max_lambda = len(lambdas)

    print('Poly. grad. =', min_order, max_order)
    print('lambda = ', min_lambda, max_lambda)
    print(np.shape(MSE_degree_lambda[min_order:max_order, min_lambda:max_lambda]))

    sns.heatmap(MSE_degree_lambda[min_order:max_order, min_lambda:max_lambda].T, cmap="RdYlGn_r",
    xticklabels=[str(deg) for deg in range(min_order, max_order)],
    yticklabels=[str(f'{lam:1.1E}') for lam in lambdas[min_lambda:max_lambda]],
    annot=True, annot_kws={"size": 12},
    fmt="1.4f", linewidths=1, linecolor=(30/255,30/255,30/255,1),
    cbar_kws={"orientation": "horizontal", "shrink":0.8, "aspect":40, "label": "MSE", "pad":0.05})
    ax.set_xlabel("Polynomial degree", labelpad=10)
    ax.set_ylabel(r'$\log_{10} \lambda$', labelpad=10)
    if bootstrap:
        ax.set_title(r'\bf{MSE Heatmap}' + f' - {method} (BS)')
    if CV:
        ax.set_title(r'\bf{MSE Heatmap}' + f' - {method} (CV)')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    if save and bootstrap:
        plt.savefig(f'plots/heatmap/zoom_{data}_{method}_resample_bootstrap.png', dpi=400)
        plt.clf()
    if save and CV:
        plt.savefig(f'plots/heatmap/zoom_{data}_{method}_resample_CV.png', dpi=400)
        plt.clf()

""" Heatmap for OLS """
def MSE_heatmap_plot_OLS(x_, y_, z_, order, seed, scale, bootstrap, n_boostraps, CV, k, data, zoom_low, zoom_high, save):
    mse = np.zeros(order).reshape(-1,1)
    for i in range(order):
        mse[i,0] = OLS_Reg(x_, y_, z_, i, seed, scale, bootstrap, n_boostraps, CV, k)[1]

    fig, ax = plt.subplots(figsize=(16,4))
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 40 # Space between the title and graph

    sns.heatmap(mse[int(order*zoom_low):int(order*zoom_high)].T, cmap="RdYlGn_r",
    annot=True, annot_kws={"size": 12}, #20
    xticklabels=[str(deg) for deg in range(int(order*zoom_low),int(order*zoom_high))], yticklabels='',
    fmt="1.4f", linewidths=1, linecolor=(30/255,30/255,30/255,1),
    cbar_kws={"orientation": "horizontal", "shrink":0.8, "aspect":40, "label": "MSE"})
    ax.set_xlabel("Polynomial degree", labelpad=10)
    ax.set_ylabel("MSE", labelpad=10)
    if scale:
        ax.set_title(r'\bf{MSE Heatmap - OLS} Scaled')
    if not scale:
        ax.set_title(r'\bf{MSE Heatmap - OLS} Not Scaled')
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/heatmap/zoom_{data}_OLS_scale{str(scale)}.png', dpi=400)
        plt.clf()
