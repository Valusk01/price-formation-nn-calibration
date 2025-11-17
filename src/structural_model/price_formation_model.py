import numpy as np

class PriceFormationModel:
    """
    effp_t = effp_{t-1} + sigma_e * eps_t
    midq_t = midq_{t-1} + d(effp_t - midq_{t-1}) + sigma_eta * eta_t
    """
    def __init__(self, theta: dict):
        """
        Model Parameters
        theta: dict 
            {
            "sigma_e": float, # efficient price innovation volatility
            "sigma_eta": float, # microstructure noise volatility
            "delta": float, # adjustment speed 
            }
        """
        self.theta = theta
    def simulator(self, T: int, rng: np.random.Generator) -> dict:
        """
        Simulate N paths of length T.
        Returns dictionary containing the efficient price, midquote and midquote returns series:
        {
            "efficient_price": np.ndarray # shape (T,N)
            "midquote": np.ndarray # shape (T,N)
            "returns": np.ndarray # shape (T-1,N)
        }
        """
        
        sigma_e = np.asarray(self.theta["sigma_e"], dtype = np.float64).reshape(-1)
        sigma_eta = np.asarray(self.theta["sigma_eta"], dtype = np.float64).reshape(-1)
        d = np.asarray(self.theta["delta"], dtype = np.float64).reshape(-1)
        N = sigma_e.size

        effp = np.empty((T,N), dtype = np.float64)
        midq = np.empty((T,N), dtype = np.float64)
        eps = rng.normal(loc=0, scale=1, size=(T,N))
        eta = rng.normal(loc=0, scale=1, size= (T,N))

        acc = np.zeros(N, dtype = np.float64)
        mq_prev = np.zeros(N, dtype = np.float64)

        for t in range(T):
            acc += sigma_e * eps[t]
            effp[t] = acc
            mq_prev += d * (acc - mq_prev) + sigma_eta * eta[t]
            midq[t] = mq_prev 
        
        mq_ret = np.diff(midq, axis=0)
        
        return {
            "efficient_price": effp, 
            "midquote": midq,
            "returns": mq_ret,
        }