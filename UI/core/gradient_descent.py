from typing import Callable, Dict, Optional
import numpy as np
import torch

# Physical bounds for beam geometry
BETA_MIN, BETA_MAX = 10.0, 40.0
L_MIN,    L_MAX    = 20.0, 35.0
W_MIN,    W_MAX    =  1.0,  2.0


def inverse_design(
    model,
    scaler_x,
    scaler_y,
    device,
    target_disp:      float,
    delta_temp:       float,
    E:                float = 71000.0,
    alpha:            float = 23e-6,
    lambda_disp:      float = 1.0,
    lambda_stress:    float = 0.001,
    lambda_vol:       float = 0.001,
    n_restarts:       int   = 10,
    n_iter:           int   = 2000,
    lr:               float = 0.05,
    disp_tol:         float = 0.005,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    fixed_vals = np.array([delta_temp, E, alpha], dtype="float32")

    # Compute scaled geometry bounds once
    bounds_phys = np.array(
        [[BETA_MIN, L_MIN, W_MIN, delta_temp, E, alpha],
         [BETA_MAX, L_MAX, W_MAX, delta_temp, E, alpha]],
        dtype="float32",
    )
    bounds_scaled = scaler_x.transform(bounds_phys)
    geom_min = torch.tensor(bounds_scaled[0, :3], device=device)
    geom_max = torch.tensor(bounds_scaled[1, :3], device=device)

    target_disp_scaled = torch.tensor(
        float(scaler_y.transform([[target_disp, 0.0, 0.0]])[0, 0]),
        device=device,
    )

    all_results = []

    for restart in range(n_restarts):
        geom_raw = np.array([
            np.random.uniform(BETA_MIN, BETA_MAX),
            np.random.uniform(L_MIN,    L_MAX),
            np.random.uniform(W_MIN,    W_MAX),
        ], dtype="float32")
        full_scaled = scaler_x.transform(
            np.concatenate([geom_raw, fixed_vals]).reshape(1, -1)
        )[0]

        geom_scaled  = torch.tensor(full_scaled[:3], requires_grad=True, device=device)
        fixed_scaled = torch.tensor(full_scaled[3:], requires_grad=False, device=device)
        optim = torch.optim.Adam([geom_scaled], lr=lr)

        for _ in range(n_iter):
            optim.zero_grad()
            geom_c = torch.clamp(geom_scaled, geom_min, geom_max)
            y_pred = model(torch.cat([geom_c, fixed_scaled]).unsqueeze(0)).squeeze(0)
            loss = (
                lambda_disp   * (y_pred[0] - target_disp_scaled) ** 2
                + lambda_stress * y_pred[1]
                + lambda_vol    * y_pred[2]
            )
            loss.backward()
            optim.step()

        with torch.no_grad():
            geom_c   = torch.clamp(geom_scaled, geom_min, geom_max)
            x_in     = torch.cat([geom_c, fixed_scaled]).unsqueeze(0)
            y_pred_s = model(x_in).squeeze(0).cpu().numpy()
            x_in_cpu = x_in.cpu().numpy()

        y_phys   = scaler_y.inverse_transform(y_pred_s.reshape(1, -1))[0]
        x_phys   = scaler_x.inverse_transform(x_in_cpu)[0]
        disp_err = abs(y_phys[0] - target_disp)

        result = dict(
            beta             = float(x_phys[0]),
            beam_length      = float(x_phys[1]),
            beam_width       = float(x_phys[2]),
            pred_displacement= float(y_phys[0]),
            pred_stress      = float(y_phys[1]),
            pred_vol         = float(y_phys[2]),
            disp_err         = disp_err,
            target_disp      = target_disp,
        )
        all_results.append(result)

        if progress_callback:
            progress_callback(restart + 1, n_restarts, result)

    good = [r for r in all_results if r["disp_err"] <= disp_tol]
    if good:
        best = min(good, key=lambda r: r["pred_stress"])
    else:
        best = min(all_results, key=lambda r: r["disp_err"])

    return best
