import numpy as np
from Camera.StereoCamera import StereoCamera


def run_mapping_ekf(
    features,
    w_T_i,
    cam: StereoCamera,
    selected_idx,
    obs_noise=2.0,
    init_var_xy=5.0,
    init_var_z=20.0,
):
    Nt = features.shape[0]
    M = len(selected_idx)

    mu = np.full((M, 3), np.nan)
    Sigma = np.zeros((M, 3, 3))
    initialized = np.zeros(M, dtype=bool)
    R = (obs_noise ** 2) * np.eye(4)

    Sigma = np.diag(np.array([init_var_xy, init_var_xy, init_var_z]))
    Sigma = np.expand_dims(Sigma, axis=0)  # (3, 3) -> (1, 3, 3)
    Sigma = np.repeat(Sigma, M, axis=0)  # (M, 3, 3)

    for t in range(Nt):
        feats_t = features[t]
        valid = np.all(feats_t[selected_idx] >= 0, axis=1)
        wTi = w_T_i[t]
        for j, feat_idx in enumerate(selected_idx):
            if not valid[j]:
                continue
            z = features[t, feat_idx]
            if not initialized[j]:
                m0 = cam.initialize_landmark(z, wTi)
                if m0 is None:
                    continue
                mu[j] = m0
                initialized[j] = True
                continue

            z_hat, H = cam.observation_model(mu[j], wTi)
            if z_hat is None:
                continue
            S = H @ Sigma[j] @ H.T + R
            K_gain = Sigma[j] @ H.T @ np.linalg.inv(S)
            innovation = z - z_hat
            mu[j] = mu[j] + K_gain @ innovation
            Sigma[j] = (np.eye(3) - K_gain @ H) @ Sigma[j]

    return mu, Sigma, initialized
