import numpy as np
import pywt

v = 1e-12 # Positive constant to avoid division by zero

# ---FUNCTION HELPERS---

# Variance ratio
def var_ratio(h, m, rv1, dtype):
    rh = m[h:] - m[:-h]              # h-step returns
    var_rh = rh.var(axis=0, ddof=1)  # var of h-step returns
    denom  = np.maximum(h * rv1, v)  # h * var(1-step)
    return (var_rh / denom).astype(dtype)


# moving average variance
def moving_avg_var(r, k, dtype):
    cs = np.cumsum(r, axis=0)
    ma = (cs[k:] - cs[:-k]) / float(k)
    return ma.var(axis=0, ddof=1).astype(dtype)


# autocorrelations and covariances
def acf_and_cov(x, k, dtype):
    """Autocorrelation and covariance at lag k for each path (columns)."""
    x0 = x[:-k]   # t
    xk = x[k:]    # t+k

    mu0 = np.mean(x0, axis=0)
    muk = np.mean(xk, axis=0)

    num = np.mean((x0 - mu0) * (xk - muk), axis=0)
    den = np.sqrt(x0.var(axis=0, ddof=1) * xk.var(axis=0, ddof=1))
    acf = (num/den).astype(dtype)

    cov = np.mean((x0-mu0) * (xk-muk), axis=0).astype(dtype)
    return acf, cov


# Normalized variance of second and third difference
def delta_var(m, rv1, g, dtype):
    """
    Normalized variance of second difference at step g:
    Var(m_{t+2g} - 2 m_{t+g} + m_t) / (2 g * RV1)
    Normalized variance of third difference at step g:
    Var(m_{t+3g} - 3 m_{t+2g} + 3 m_{t+g} - m_t) / (6 g * RV1)
    """
    T, N = m.shape

    acc2_norm = np.zeros(N, dtype=dtype)
    acc3_norm = np.zeros(N, dtype=dtype)

    # 2nd difference: m_{t+2g} - 2 m_{t+g} + m_t
    d2 = m[2*g:] - 2.0 * m[g:-g] + m[:-2*g]   # shape (T-2g, N)
    var2 = d2.var(axis=0, ddof=1)
    denom2 = 2.0 * g * np.maximum(rv1, v)
    acc2_norm = (var2 / denom2).astype(dtype)

    # 3rd difference: m_{t+3g} - 3 m_{t+2g} + 3 m_{t+g} - m_t
    
    d3 = (m[3*g:] - 3.0 * m[2*g:-g] + 3.0 * m[g:-2*g] - m[:-3*g])  # shape (T-3g, N)
    var3 = d3.var(axis=0, ddof=1)
    denom3 = 6.0 * g * np.maximum(rv1, v)
    acc3_norm = (var3 / denom3).astype(dtype)

    return acc2_norm, acc3_norm


# returns in the frequency domain: spectral tilt and bandpowers
def spectral_and_bandpowers(returns, bands, K_tapers, dtype):
    """
    Compute:
      - spectral tilt (HF / LF) using a single FFT
      - multitaper bandpowers (HF / MID / LF, etc.) using sine tapers.

    Parameters
    ----------
    returns : (T, N) array
        Return series (time × paths).
    bands   : list/tuple of (low, high)
        Frequency bands in cycles per sample, e.g. [(0.15,0.25), (0.075,0.15), ...].
    dtype   : numpy dtype for outputs (e.g. np.float64)
    K_tapers: int
        Number of sine tapers for multitaper bandpowers.

    Returns
    -------
    tilt      : (N,) array
        Spectral tilt = HF power / LF power per path.
    band_pows : list of (N,) arrays
        One array per band, multitaper-averaged bandpower.
    """

    T, N = returns.shape
    F = np.fft.rfft(returns, axis=0) # complex valued fourier coefficients for non-negative frequencies only - spectrum per path
    P = (F * np.conjugate(F)).real # power spectrum - variance contribution

    P = P[1:] # P[0] corresponds to frequency 0, the mean, we care about fluctuations
    K = P.shape[0] # number of rows in P
    k20 = max(1, int(0.2 * K)) 

    lf_power = P[:k20].mean(axis=0) # mean of lowest 20% of frequencies for each path
    hf_power = P[-k20:].mean(axis=0) # mean of highest 20%
    # average variance contribution from low and high frequencies
    
    tilt = hf_power / np.maximum(lf_power, v)
    # tilt > 1 -> path j has more high frequency variability (more noise)
    # tilt < 1 -> most variance at low frequencies (slow movements)
    # tilt = 1 -> similar HF and LF
    
    #  Multitaper bandpowers over specified bands
    freqs = np.fft.rfftfreq(T)
    n_bands = len(bands)

    bp_acc = np.zeros((n_bands, N), dtype=np.float64) # accumulator for bandpowers: (n_bands, N)

    # build sine tapers
    n = np.arange(1, T + 1, dtype=np.float64)          
    norm = np.sqrt(2.0 / (T + 1))

    for k in range(1, K_tapers + 1):
        taper = norm * np.sin(np.pi * k * n / (T + 1))  # (T,)
        xk = returns * taper[:, None]                   # apply taper to all paths

        Fk = np.fft.rfft(xk, axis=0)                    # (F_len, N)
        Pk = (Fk * np.conjugate(Fk)).real               # power

        for bi, (low, high) in enumerate(bands):
            idx = (freqs >= low) & (freqs <= high)
            if idx.any():
                bp_acc[bi] += Pk[idx].mean(axis=0)      # add mean power in band

    # average across tapers
    bp_acc /= float(K_tapers)
    band_pows = [bp_acc[bi].astype(dtype) for bi in range(n_bands)]
    
    return tilt, band_pows


# haar wavelet variances
def haar_detail_vars(returns, dtype):
    """
    Repeatedly smoothing and differencing the return series at increasing scales, 
    then using the variances of those detail coefficients as moments. 
    Explain how volatility is distributed across short, medium, and slightly longer time scales.
    """
    T, N = returns.shape
    v1 = np.zeros(N, dtype=dtype)
    v2 = np.zeros(N, dtype=dtype)
    v3 = np.zeros(N, dtype=dtype)
    v4 = np.zeros(N, dtype=dtype)

    wave = pywt.Wavelet("haar")
    
    for j in range(N):
        series = np.asarray(returns[:,j], dtype=float)
        # Maximum possible level for this length & wavelet
        max_level = pywt.dwt_max_level(series.size, wave.dec_len)
        
        # up to level 4 
        level = min(4, max_level)

        # wavedec returns: [cA_level, cD_level, cD_{level-1}, ..., cD1]
        coeffs = pywt.wavedec(series, wavelet=wave, level=level)
        # details: [cD_level, ..., cD1]
        details = coeffs[1:]

        # cD1 (finest scale) is the *last* detail
        
        cD1 = details[-1]
        v1[j] = np.var(cD1, ddof=1)

        # cD2
        cD2 = details[-2]
        v2[j] = np.var(cD2, ddof=1)

        # cD3
        cD3 = details[-3]
        v3[j] = np.var(cD3, ddof=1)

        # cD4
        cD4 = details[-4]
        v4[j] = np.var(cD4, ddof=1)

    return v1.astype(dtype), v2.astype(dtype), v3.astype(dtype), v4.astype(dtype)


# Variance at difference horizons and ols regression of var on the horizon
def varh_and_ols_ts_slopes(midquote, h_list, dtype):
    """
    For each horizon h in h_list, compute Var(m_{t+h} - m_t) per path,
    then fit Var_h ≈ a + b * h by OLS and Theil-Sen, per path.
    Parameters
    ----------
    midquote : (T, N) array
    h_list   : iterable of ints (horizons)
    dtype    : np.float64, etc.

    Returns
    -------
    a : (N,)  OLS intercept per path
    b : (N,)  OLS slope per path
    ts_slope : (N,) array Theil Sen slope per path.
    var_h_matrix : (H, N) matrix of variances at each horizo
    """
    m = np.asarray(midquote, dtype=dtype)
    T, N = m.shape

    h = np.asarray(h_list, dtype=np.int64)  
    H = h.size
    var_list = []

    # compute Var(m_{t+h} - m_t) for each h 
    for hi in h:
        rh = m[hi:] - m[:-hi]  # h-step increments, shape (T-hi, N)
        var_list.append(rh.var(axis=0, ddof=1).astype(dtype))

    var_h_matrix = np.stack(var_list, axis=0)  

    # OLS fit Var_h ≈ a + b*h (per path)
    h_mean = h.mean()
    h_centered = h - h_mean                   

    # covariance between h and Var_h, per path
    cov_hy = (h_centered[:, None] * (var_h_matrix - var_h_matrix.mean(axis=0))).mean(axis=0)  
    var_hh = h_centered.var() + v                # scalar, avoid /0

    b = cov_hy / var_hh                          # slope per path
    a = var_h_matrix.mean(axis=0) - b * h_mean   # intercept per path

    # Theil–Sen slope (robust) per path
    ts_slope = np.zeros(N, dtype=dtype)
    h_float = h.astype(np.float64)

    for j in range(N):
        y = var_h_matrix[:, j].astype(np.float64)
        slopes = []
        for i in range(H):
            for k in range(i + 1, H):
                dh = h_float[k] - h_float[i]
                dy = y[k] - y[i]
                slopes.append(dy / dh)
        
        ts_slope[j] = np.median(slopes)

    return a.astype(dtype), b.astype(dtype), ts_slope.astype(dtype), var_h_matrix


# decimated features and contrasts
def decimated_features(midquote, h_vr, suffix, dtype):
    """
    Compute VR/ACF/MA moments on a decimated midquote series m[::d].

    m          : (T_d, N) decimated midquote
    h_vr_list  : horizons for VR (e.g. [10, 50])
    suffix     : string for names (e.g. 'D5', 'D2', ...)
    """
    feats_d, names_d = [], []

    # returns and variance on the decimated grid
    r1d = np.diff(midquote, axis=0)
    rv1d = r1d.var(axis=0, ddof=1).astype(dtype)

    # variance ratios on the decimated series
    h10, h50 = h_vr
    VR10_d = var_ratio(h10, midquote, rv1d, dtype)
    VR50_d = var_ratio(h50, midquote, rv1d, dtype)

    feats_d.append(VR10_d); names_d.append(f"VR10_{suffix}")
    feats_d.append(VR50_d); names_d.append(f"VR50_{suffix}")

    slope_10_50_d = (VR50_d - VR10_d) / float(h50 - h10) 
    feats_d.append(slope_10_50_d); names_d.append(f"VR_SLOPE_10_50_{suffix}")

    # VR area deficit on decimated grid
    area_d = np.zeros(rv1d.shape, dtype=dtype)
    for h in range (5, 51, 5):
        area_d += (1.0 - var_ratio(h, midquote, rv1d, dtype))
    feats_d.append(area_d); names_d.append(f"VR_AREA_DEF_5_50_{suffix}")

    # ACF on decimated returns 
    acf_d = {}
    for lag in range(1, 6):
        rho_l, _ = acf_and_cov(r1d, lag, dtype)
        acf_d[lag] = rho_l
        feats_d.append(rho_l); names_d.append(f"ACF_LAG{lag}_{suffix}")

    # moving average on decimated returns
    ma_over_d = {}
    for k in [3, 5, 8]:
        var_ma_d = moving_avg_var(r1d, k, dtype)
        feats_d.append(var_ma_d); names_d.append(f"MA{k}_VAR_{suffix}")

        ma_over = np.where(rv1d == 0.0, 0.0, var_ma_d / rv1d)
        feats_d.append(ma_over); names_d.append(f"MA{k}_VAR_over_RV1_{suffix}")
        ma_over_d[k] = ma_over

    return feats_d, names_d, slope_10_50_d, acf_d, ma_over_d       

# ---MOMENT FUNCTION---
def realized_moments(midquote, returns, dtype):
    """
    Compute realized moments per path using midquotes and midquote returns.

    Returns

    X : (N, n_features) array
        Feature matrix, one row per path.
    names : list of str
        Feature names corresponding to columns of X.
    """
    feats, names = [], []
    m  = np.asarray(midquote, dtype=dtype)
    r1 = np.asarray(returns, dtype=dtype)
    rv1 = r1.var(axis=0, ddof=1).astype(dtype)
    

    # --SCALE AND DISPERSION FEATURES--
    feats.append(rv1); names.append("Realized_Variance1")

    med_r1 = np.median(r1, axis=0)
    mad_r1 = np.median(np.abs(r1 - med_r1), axis=0).astype(dtype)
    q25, q75 = np.percentile(r1, [25, 75], axis=0)
    iqr_r1 = (q75 - q25).astype(dtype)

    feats.append(mad_r1.astype(dtype)); names.append("MAD_r1")
    feats.append(iqr_r1.astype(dtype)); names.append("IQR_r1")

    denom = np.sqrt(np.maximum(rv1, v))
    feats.append((mad_r1 / denom).astype(dtype)); names.append("MAD_over_sqrtRV1")
    feats.append((iqr_r1 / denom).astype(dtype)); names.append("IQR_over_sqrtRV1")    


    # --VARIANCE RATIO FEATURES--
    VR10  = var_ratio(10, m, rv1, dtype); feats.append(VR10); names.append("VR10")
    VR20 = var_ratio(20, m, rv1, dtype);feats.append(VR20); names.append("VR20")
    VR30 = var_ratio(30, m, rv1, dtype);feats.append(VR30); names.append("VR30")
    VR50  = var_ratio(50, m, rv1, dtype);feats.append(VR50); names.append("VR50")
    VR100 = var_ratio(100, m, rv1, dtype);feats.append(VR100); names.append("VR100")

    # slope between 10-50 and 50-100
    base_vr_slope_10_50 = (VR50 - VR10) / 40.0
    feats.append(base_vr_slope_10_50); names.append("VR_SLOPE_10_50")
    feats.append((VR100 - VR50) / 50.0); names.append("VR_SLOPE_50_100")
    
    # how much VR deviates from 1
    area = np.zeros(rv1.shape, dtype=dtype)
    for h in range(5, 51, 5):
        area += (1.0 - var_ratio(h, m, rv1, dtype))
    feats.append(area); names.append("VR_AREA_DEF_5_50")
    
    # second finite difference
    feats.append((VR50 - 2.0 * VR30 + VR10).astype(dtype))
    names.append("VR_CURV_10_30_50")


    # --MOVING AVERAGE FEATURES--
    ma_over_base = {}
    for k in [3, 5, 8]:
        var_ma = moving_avg_var(r1, k, dtype)     # variance of k-step moving average
        feats.append(var_ma); names.append(f"MA{k}_VAR")

        ma_over = var_ma / rv1
        feats.append(ma_over); names.append(f"MA{k}_VAR_over_RV1")
        ma_over_base[k] = ma_over

        resid = np.maximum(rv1 - var_ma, 0.0)
        feats.append(resid); names.append(f"MA{k}_RESID_VAR")

        resid_over = resid / rv1
        feats.append(resid_over); names.append(f"MA{k}_RESID_VAR_over_RV1")


    # --AUTOCORRELATION AND COVARIANCE FEATURES--
    rhos = []
    acf_base = {}

    for lag in range(1, 6):
        rho_l, cov_l = acf_and_cov(r1, lag, dtype=dtype)
        rhos.append(rho_l)
        acf_base[lag] = rho_l
        feats.append(rho_l); names.append(f"ACF_LAG{lag}")
        feats.append(cov_l); names.append(f"COV_LAG{lag}")

    tot_neg_acfs = np.zeros_like(rhos[0], dtype=dtype)
    for rho_l in rhos:
        tot_neg_acfs += np.maximum(-rho_l,0.0)
    feats.append(tot_neg_acfs); names.append("TOT_NEG_ACFS_lag1_5")


    # --NORMALIZED VARIANCES OF SECOND AND THIRD DIFFERENCES--
    for g in [1,2,4,6]:
        acc2, acc3 = delta_var(m, rv1, g, dtype=dtype)
        feats.append(acc2); names.append(f"ACCVAR_NORM_g{g}")
        feats.append(acc3); names.append(f"ACC3VAR_NORM_g{g}")
        
        acc3_over_acc2 = (acc3 / np.maximum(acc2, v)).astype(dtype)
        feats.append(acc3_over_acc2); names.append(f"ACC3_OVER_ACC2_RATIO_g{g}")


    # --SPECTRAL FEATURES--
    bands =  [(0.15, 0.25), (0.075, 0.15), (0.01, 0.04)]  # frequency ranges (fast - mid - slow variation)
    
    tilt, band_pows = spectral_and_bandpowers(r1, bands, 3, dtype=dtype)  
    feats.append(tilt); names.append("SPECTRAL_TILT")

    # log1p of bandpowers
    bp_hf, bp_mid, bp_lf = band_pows
    feats.append(np.log1p(bp_hf));  names.append("BP_HF_log1p")
    feats.append(np.log1p(bp_mid)); names.append("BP_MID_log1p")
    feats.append(np.log1p(bp_lf));  names.append("BP_LF_log1p")

    # ratios HF/MID and HF/LF
    feats.append(bp_hf / np.maximum(bp_mid, v)); names.append("BP_HF_over_MID")
    feats.append(bp_hf / np.maximum(bp_lf,  v)); names.append("BP_HF_over_LF")


    # --HAAR WAVELET FEATURES--
    v1, v2, v3, v4 = haar_detail_vars(r1, dtype=dtype)   # each (N,)

    # normalize by RV1 and log1p
    v1_norm = (v1 / np.maximum(rv1, v))
    v2_norm = (v2 / np.maximum(rv1, v))
    v3_norm = (v3 / np.maximum(rv1, v))
    v4_norm = (v4 / np.maximum(rv1, v))

    feats.append(np.log1p(v1_norm)); names.append("HAAR_v1_over_RV1_log1p")
    feats.append(np.log1p(v2_norm)); names.append("HAAR_v2_over_RV1_log1p")
    feats.append(np.log1p(v3_norm)); names.append("HAAR_v3_over_RV1_log1p")
    feats.append(np.log1p(v4_norm)); names.append("HAAR_v4_over_RV1_log1p")

    # scale ratios
    v1_over_v2 = (v1 / np.maximum(v2, v))
    v1_over_v3 = (v1 / np.maximum(v3, v))
    v1_over_v4 = (v1 / np.maximum(v4,v))

    feats.append(v1_over_v2.astype(dtype)); names.append("HAAR_v1_over_v2")
    feats.append(v1_over_v3.astype(dtype)); names.append("HAAR_v1_over_v3")
    feats.append(v1_over_v4.astype(dtype)); names.append("HAAR_v1_over_v4")


    # --VAR VS HORIZON FEATURES--
    h = [10, 20, 30, 50]
    a, b, ts_slope, var_h_matrx = varh_and_ols_ts_slopes(m, h, dtype=dtype)

    loga = np.log(np.maximum(a, v)).astype(dtype)
    feats.append(loga);        names.append("LOG_a")
    feats.append(0.5 * loga);  names.append("HALF_LOG_a")

    feats.append(b.astype(dtype)); names.append("VARFIT_SLOPE_10_50")

    var10 = var_h_matrx[0]
    var20 = var_h_matrx[1]
    var30 = var_h_matrx[2]
    var_curv_10_20_30 = (var30 - 2.0 * var20 + var10).astype(dtype)
    feats.append(var_curv_10_20_30); names.append("VAR_CURV_10_20_30")
    feats.append(ts_slope.astype(dtype)); names.append("VARFIT_TS_SLOPE_10_50")


    # --DELTA PROXY & HALF LIFE FEATURES--
    EPS = 1e-12

    rho1 = rhos[0]
    rho2 = rhos[1]
    rho3 = rhos[2]

    ratio21 = rho2 / rho1
    ratio31 = rho3 / rho1

    safe_ratio21 = np.maximum(np.abs(ratio21), v)
    safe_ratio31 = np.maximum(np.abs(ratio31), v)

    delta_hat21 = -np.log(safe_ratio21)   
    delta_hat31 = -np.log(safe_ratio31)   

    # Second log to compress range
    delta_hat21_log = np.log(np.maximum(delta_hat21, v))
    delta_hat31_log = np.log(np.maximum(delta_hat31, v))

    feats.append(delta_hat21_log.astype(dtype)); names.append("LOG_deltaHat21")
    feats.append(delta_hat31_log.astype(dtype)); names.append("LOG_deltaHat31")
    feats.append(0.5 * delta_hat21_log.astype(dtype)); names.append("HALF_LOG_deltaHat21")
    feats.append(0.5 * delta_hat31_log.astype(dtype)); names.append("HALF_LOG_deltaHat31")

    # Half-life proxies: 1/delta
    hl_proxy21 = np.where(delta_hat21 <= v, 0.0, 1.0 / np.maximum(delta_hat21, v))
    hl_proxy31 = np.where(delta_hat31 <= v, 0.0, 1.0 / np.maximum(delta_hat31, v))

    feats.append(hl_proxy21.astype(dtype)); names.append("HL_PROXY21")
    feats.append(hl_proxy31.astype(dtype)); names.append("HL_PROXY31")

    proxy_a_sqrt_21 = a * np.sqrt(np.maximum(delta_hat21, 0.0))
    proxy_a_sqrt_31 = a * np.sqrt(np.maximum(delta_hat31, 0.0))

    feats.append(proxy_a_sqrt_21.astype(dtype)); names.append("PROXY_a_sqrtNegLogRatio_21")
    feats.append(proxy_a_sqrt_31.astype(dtype)); names.append("PROXY_a_sqrtNegLogRatio_31")


    # --DECIMATED FEATURES AND CONTRASTS--
    # horizons for VR on decimated grid
    h_vr_list = [10, 50]

    # decimation factors
    DECIMS = [2, 5, 8]

    # atanh for ACF contrasts
    def atanh_clip(x):
        return np.arctanh(np.clip(x, -0.99, 0.99))

    for d in DECIMS:
        md = m[::d, :]   # (T_d, N)
        suffix = f"D{d}"

        # moments on the decimated series
        feats_d, names_d, slope_d, acf_d, ma_over_d = decimated_features(
            md, h_vr=h_vr_list, dtype=dtype, suffix=suffix
        )

        feats.extend(feats_d)
        names.extend(names_d)

        # contrasts: VR slope (decim d) - (base)
        vr_slope_contr = (slope_d - base_vr_slope_10_50).astype(dtype)
        feats.append(vr_slope_contr)
        names.append(f"VR_SLOPE_10_50_CONTR_D{d}_minus_D1")

        # contrasts: ACF (atanh) decim d - base
        for lag in range(1, 6):
            base_acf = acf_base[lag]
            dec_acf  = acf_d[lag]

            atanh_base = atanh_clip(base_acf)
            atanh_dec  = atanh_clip(dec_acf)

            acf_contr = (atanh_dec - atanh_base).astype(dtype)
            feats.append(acf_contr)
            names.append(f"ACFCONTR_L{lag}_D{d}_minus_D1")

        # contrasts: MAk_VAR_over_RV1 decim d - base
        for k in [3, 5, 8]:
            base_ma_over = ma_over_base[k]
            dec_ma_over  = ma_over_d[k]
            ma_contr = (dec_ma_over - base_ma_over).astype(dtype)
            feats.append(ma_contr)
            names.append(f"MA{k}_VAR_over_RV1_CONTR_D{d}_minus_D1")


    X = np.stack(feats, axis=1).astype(dtype)
    return X, names
    
