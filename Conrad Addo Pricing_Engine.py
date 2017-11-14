import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict
import scipy.interpolate as interp
import scipy.stats as sci
from scipy.stats import norm
from scipy import interpolate
from numpy import array
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')


class Conrad:
    """Vanna-Volga Implied Volatility method used to price an FX option"""

    def __init__(self, S0, strike, t, importfile):
        self.dc = 360  # day count convention
        self.S0 = S0
        self.strike = strike
        self.t = t
        self.importfile = importfile

    def foreign_rate(self, fp, Rates):
        xf = [] * len(fp)
        Rates_array = np.array(Rates).squeeze()
        Rates_interp = interp.interp1d(np.arange(Rates_array.size), Rates_array, kind='cubic')
        Rates_compress = Rates_interp(np.linspace(0, Rates_array.size - 1, fp.size))
        for i in range(len(Rates_compress)):
            xf.append(((fp[i] / self.S0) + 1) * (1 + Rates_compress[i]) - 1)

        return Rates_compress, xf

    def k25_data(self, T, sig25DeltaPut, sigATM, sig25DeltaCall, xd, xf):
        """Calculate the strikes K25DeltaPut, KATM, K25DeltaCall"""

        # Pricing Inputs
        Dd = np.exp(-np.multiply(xd, T))
        Df = np.exp(-np.multiply(xf, T))
        delta = 0.25
        alpha = -sci.norm.ppf(delta * np.reciprocal(Df))
        mu = xd - xf
        F = self.S0 * np.exp(np.multiply(mu, T))

        # Strikes
        KATM = np.array([])
        K25DeltaCall = np.array([])
        K25DeltaPut = np.array([])

        for i in range(len(T)):
            K25DeltaPut = np.append(K25DeltaPut, F[i] * math.exp(
                -(alpha[i] * sig25DeltaPut[i] * math.sqrt(T[i])) + (xd[i] - xf[i] + 0.5 * sig25DeltaPut[i] ** 2) * T[i]))
            KATM = np.append(KATM, F[i] * math.exp(xd[i] - xf[i] + 0.5 * T[i] * (sigATM[i]) ** 2))
            K25DeltaCall = np.append(K25DeltaCall, F[i] * math.exp(
                alpha[i] * sig25DeltaCall[i] * math.sqrt(T[i]) + (xd[i] - xf[i] + 0.5 * sig25DeltaCall[i] ** 2) * T[i]))

        return K25DeltaPut, KATM, K25DeltaCall, F

    def d1(self, F, K, sig, T, xd, xf):
        d1_val = (np.log(F / K) + (xd - xf + 0.5 * (sig ** 2)) * T) / (sig * np.sqrt(T))

        return d1_val

    def d2(self, F, K, sig, T, xd, xf):
        d2_val = self.d1(F, K, sig, T, xd, xf) - sig * np.sqrt(T)

        return d2_val

    def bs_price(self, F, K, T, xd, xf, sig):
        a = norm.cdf(self.d1(F, K, sig, T, xd, xf))
        b = self.d1(F, K, sig, T, xd, xf)

        bs_val = norm.cdf(self.d1(F, K, sig, T, xd, xf)) * F * math.exp(-xf * T) - norm.cdf(
            self.d2(F, K, sig, T, xd, xf)) * F * math.exp(-xd * T)

        return bs_val

    def vega(self, F, K, T, xd, xf, sig):
        np = math.exp(-self.d1(F, K, sig, T, xd, xf) ** 2 / 2) / math.sqrt(2 * math.pi)
        vega_val = (F * math.exp(-xf * T) * math.sqrt(T) * np) / 100

        return vega_val

    def delta(self, F, K, T, xd, xf, sig):
        delta_val = math.exp(xf * T) * norm.cdf(self.d1(F, K, sig, T, xd, xf))

        return delta_val

    def implied_price(self, F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf):
        v_v1 = self.vega(F, K, T, xd, xf, sig2) / self.vega(F, K1, T, xd, xf, sig1)
        v_v2 = self.vega(F, K, T, xd, xf, sig2) / self.vega(F, K2, T, xd, xf, sig2)
        v_v3 = self.vega(F, K, T, xd, xf, sig2) / self.vega(F, K3, T, xd, xf, sig3)

        x1 = v_v1 * ((math.log(K2 / K) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K1)))
        x2 = v_v2 * ((math.log(K / K1) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K2)))
        x3 = v_v3 * ((math.log(K / K1) * math.log(K / K2)) / (math.log(K3 / K1) * math.log(K3 / K2)))

        a = self.bs_price(F, K2, T, xd, xf, sig2)

        vv_price = self.bs_price(F, K2, T, xd, xf, sig2) + x1 * (
        self.bs_price(F, K1, T, xd, xf, sig1) - self.bs_price(F, K1, T, xd, xf, sig2))
        + x3 * (self.bs_price(F, K3, T, xd, xf, sig3) - self.bs_price(F, K3, T, xd, xf, sig2))

        return vv_price

    def market_price(self, F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf, j):
        sig = self.implied_vol(F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf, j)
        market_val = self.bs_price(F, K, T, xd, xf, sig)

        return market_val

    def implied_vol(self, F, K, T, K1, K2, K3, sig1, sig2, sig3, xd, xf, j):
        x1 = (math.log(K2 / K) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K1))
        x2 = (math.log(K / K1) * math.log(K3 / K)) / (math.log(K2 / K1) * math.log(K3 / K2))
        x3 = (math.log(K / K1) * math.log(K / K2)) / (math.log(K3 / K1) * math.log(K3 / K2))

        D1 = (x1 * sig1 + x2 * sig2 + x3 * sig3) - sig2

        D2 = (x1 * self.d1(F, K1, sig1, T, xd, xf) * self.d2(F, K1, sig1, T, xd, xf) * (sig1 - sig2) ** 2) \
             + (x2 * self.d1(F, K2, sig2, T, xd, xf) * self.d2(F, K2, sig2, T, xd, xf) * (sig2 - sig2) ** 2) \
             + (x3 * self.d1(F, K3, sig3, T, xd, xf) * self.d2(F, K3, sig3, T, xd, xf) * (sig3 - sig2) ** 2)

        sig = sig2 + (-sig2 + np.sqrt(
            sig2 ** 2 + self.d1(F, K, sig2, T, xd, xf) * self.d2(F, K, sig2, T, xd, xf) * (2 * sig2 * D1 + D2))) \
                     / (self.d1(F, K, sig2, T, xd, xf) * self.d2(F, K, sig2, T, xd, xf))

        return sig

    def interp_vv(self, row_find, col_find, flat_row_array, flat_col_array, z_matrix):
        GD = interpolate.griddata((flat_row_array, flat_col_array), z_matrix,
                                  ([row_find], [col_find]), method='cubic')

        return GD

    def create_plots(self, F, T, K25DeltaPut, KATM, K25DeltaCall, sig25DeltaPut, sigATM, sig25DeltaCall, xd,
                     xf):
        m = len(T[:-1])
        K_array = np.linspace(0.8 * self.S0, 1.2 * self.S0, m)


        implied_matrix = np.zeros((m, m), dtype=float)
        market_price_matrix = np.zeros((m, m), dtype=float)
        price_matrix = np.zeros((m, m), dtype=float)
        delta_matrix = np.zeros((m, m), dtype=float)

        for i in range(K_array.size):
            for j in range(m):
                implied_matrix[i][j] = self.implied_vol(F[j], K_array[i], T[j], K25DeltaPut[j], KATM[j],
                                                        K25DeltaCall[j],
                                                        sig25DeltaPut[j], sigATM[j], sig25DeltaCall[j], xd[j], xf[j], j)

        implied_matrix = pd.DataFrame(implied_matrix)
        implied_matrix.fillna(method='ffill', axis=1, inplace=True)

        for i in range(K_array.size):
            for j in range(m):
                price_matrix[i][j] = self.implied_price(F[j], K_array[i], T[j], K25DeltaPut[j], KATM[j],
                                                        K25DeltaCall[j],
                                                        sig25DeltaPut[j], sigATM[j], sig25DeltaCall[j], xd[j], xf[j])
                market_price_matrix[i][j] = self.market_price(F[j], K_array[i], T[j], K25DeltaPut[j], KATM[j],
                                                              K25DeltaCall[j],
                                                              sig25DeltaPut[j], sigATM[j], sig25DeltaCall[j], xd[j],
                                                              xf[j], j)
                delta_matrix[i][j] = self.delta(F[j], K_array[i], T[j], xd[j], xf[j], implied_matrix[i][j])

        #  Fill NAs
        implied_matrix = pd.DataFrame(implied_matrix)
        price_matrix = pd.DataFrame(price_matrix)
        delta_matrix = pd.DataFrame(delta_matrix)
        implied_matrix.fillna(method='ffill', axis=1, inplace=True)
        price_matrix.fillna(method='ffill', axis=1, inplace=True)
        delta_matrix.fillna(method='ffill', axis=1, inplace=True)

        #  Flatten the matrices for use with griddata
        implied_flat = implied_matrix.as_matrix().ravel()
        price_flat = price_matrix.as_matrix().ravel()
        delta_flat = delta_matrix.as_matrix().ravel()

        KK, TT = np.meshgrid(np.array(K_array), np.array(T[:-1]), indexing='ij')

        option_vol = interpolate.griddata((KK.ravel(), TT.ravel()), implied_flat,
                                  ([self.t], [self.strike]), method='nearest')

        option_premium = interpolate.griddata((KK.ravel(), TT.ravel()), price_flat,
                                  ([self.t], [self.strike]), method='nearest')

        option_delta = interpolate.griddata((KK.ravel(), TT.ravel()), delta_flat,
                                  ([self.t], [self.strike]), method='nearest')

        #  Gridspec
        gs = gridspec.GridSpec(2, 2)

        #  3D Meshgrid of T, K and Volatility
        X = T[:-1]
        Y = K_array
        X, Y = np.meshgrid(X, Y)
        Z = 100*implied_matrix
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[:-1, :], projection='3d')
        ax1.set_xlabel('Tenor (yrs)')
        ax1.set_ylabel('Strike')
        ax1.set_zlabel('Implied Volatility (%)')
        plt.suptitle('Option Greeks')
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.gnuplot, linewidth=0)

        #  Implied Volatility vs Strikes at 3M
        X = K_array
        Y = 100*implied_matrix[:][7]
        ax2 = fig.add_subplot(gs[-1, 0])
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Implied Volatility (%)')
        plt.title('Implied Volatility vs Strikes at 3M')
        f = ax2.plot(X, Y)

        #  Option Premium vs Strikes at 3M
        X = K_array
        Y = price_matrix[:][7]
        ax3 = fig.add_subplot(gs[-1, -1])
        ax3.set_xlabel('Strike')
        ax3.set_ylabel('C(K) (USD)')
        plt.title('Option Premium vs Strikes at 3M')
        g = ax3.plot(X, Y, 'r')
        fig.set_size_inches(13, 10)
        plt.show()

        return 'Option Volatility: {}%, Option Premium: {} USD, Option delta: {}'.format(option_vol*100, option_premium, option_delta)

    def FXOptionPrice(self):

        # Import Excel file
        xlsx_file = pd.ExcelFile(self.importfile)

        # Volatility Data
        vol_data = xlsx_file.parse('Volatility')

        # US Yield Data
        swaps_futures = xlsx_file.parse('US_Yield')
        Rates = np.array(swaps_futures['Value']) / 100
        T_Rates = pd.DataFrame({'X': swaps_futures['Term']})

        # Forward Points data
        fp_data = xlsx_file.parse('Forward_Points')
        fp = np.array(fp_data['Value']) / 10000
        T_fp = pd.DataFrame({'X': fp_data['Term']})

        T = pd.DataFrame({'X': vol_data['Term'].unique()})  # terms

        # dictionary for excel tabulated terms converted to day/360 format
        super_dict = [('SWAP', ''), ('WK', '*7'), ('MO', '*30'), ('M', '*30'), ('D', '*1'), ('DY', '*1'),
                      ('W', '*7'), ('YR', '*360'), ('Y', '*360'), (' ', '')]
        super_dict = OrderedDict(super_dict)

        # Modify the terms to daycount and store the vols in numpy arrays
        T = T.replace(super_dict, regex=True)
        T_Rates = T_Rates.replace(super_dict, regex=True)
        T_fp = T_fp.replace(super_dict, regex=True)

        T = T.X.apply(lambda x: eval(str(x))) / self.dc
        T_Rates = T_Rates.X.apply(lambda x: eval(str(x))) / self.dc
        T_fp = T_fp.X.apply(lambda x: eval(str(x))) / self.dc

        sigATM = np.array(vol_data[vol_data['Strike'] == 'ATM']['Value']) / 100  # ATM volatilities
        sig25DeltaRR = np.array(
            vol_data[vol_data['Strike'] == '25 Delta Risk Reversal']['Value']) / 100  # ATM volatilities
        sig25DeltaBF = np.array(vol_data[vol_data['Strike'] == '25 Delta Butterfly']['Value']) / 100  # ATM volatilities
        sig25DeltaCall = sig25DeltaBF + sigATM + sig25DeltaRR / 2
        sig25DeltaPut = sig25DeltaBF + sigATM - sig25DeltaRR / 2

        xd, xf = self.foreign_rate(fp, Rates)
        K25DeltaPut, KATM, K25DeltaCall, F = self.k25_data(T, sig25DeltaPut, sigATM, sig25DeltaCall, xd, xf)
        option_data = self.create_plots(F, T, K25DeltaPut, KATM, K25DeltaCall, sig25DeltaPut, sigATM, sig25DeltaCall, xd, xf)

        return option_data

def main():
    # Example Input
    # FXOptionPrice(1.1609, 1.2, 30 / 360, 'Case Study 2017 - Pricing Engine.xlsx')

    validus_answer = Conrad(1.1609, 1.1, 5, 'Case Study 2017 - Pricing Engine.xlsx').FXOptionPrice()
    print(validus_answer)
    plt.show()

if __name__ == "__main__":
    main()
