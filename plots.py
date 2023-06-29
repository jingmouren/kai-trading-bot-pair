from utils import *

__author__ = 'kq'


def adf_cdf() -> None:
    """

    :return:
    """
    for file in glob.glob(HOME + 'adf/*.csv'):
        date = file.split('/')[-1].split('.')[0]
        adf = pd.read_csv(file)
        fraction_feasible = {}
        for j in [k / 20 for k in range(21)]:
            feasible = adf[adf.P_VALUE <= j]
            fraction_feasible[j] = len(feasible) / len(adf)
        pd.Series(fraction_feasible).plot(label=date)
    plt.title(r'CDF of Fraction of Feasibility by $p_{max}$')
    plt.ylabel('Fraction')
    plt.xlabel(r'$p_{max}$')
    plt.axvline(0.02, linestyle='--', color='red')
    plt.axvline(0.05, linestyle='--', color='orange')
    plt.axvline(0.10, linestyle='--', color='yellow')
    plt.legend()
