import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from core.model_loader     import load_model
from core.gradient_descent import inverse_design
from components            import model_selector, param_inputs, results_panel

st.set_page_config(
    page_title = "V-Beam Inverse Design",
    layout     = "wide",
)

st.markdown(
    """
    <style>
        .stDeployButton                   { display: none !important; }
        [data-testid="stAppDeployButton"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading model…")
def get_model(model_type: str):
    return load_model(model_type)

with st.sidebar:
    st.title("V-Beam Designer")
    st.markdown("---")

    selected_model = model_selector.render()
    st.markdown("---")

    target_disp, delta_temp = param_inputs.render(selected_model)
    st.markdown("---")

    predict_clicked = st.button(
        "PREDICT",
        type                = "primary",
        use_container_width = True,
    )

st.title("V-Beam Thermal Sensor -- Inverse Design")
st.caption(
    "Given a target sensor displacement and temperature change, "
    "gradient descent finds the optimal beam geometry (β, L, w)."
)

if predict_clicked:
    model, scaler_x, scaler_y, device = get_model(selected_model)

    with st.status("Running gradient descent…", expanded=True) as status:

        def _progress(restart_num: int, total: int, res: dict):
            err_um = res["disp_err"] * 1000
            st.write(
                f"Restart **{restart_num}/{total}** — "
                f"β = {res['beta']:.2f}°, "
                f"L = {res['beam_length']:.2f} mm, "
                f"w = {res['beam_width']:.2f} mm, "
                f"err = {err_um:.1f} µm"
            )

        result = inverse_design(
            model             = model,
            scaler_x          = scaler_x,
            scaler_y          = scaler_y,
            device            = device,
            target_disp       = target_disp,
            delta_temp        = delta_temp,
            progress_callback = _progress,
        )
        status.update(label="Optimization complete!", state="complete", expanded=False)

    st.session_state["last_result"] = result

results_panel.render(st.session_state.get("last_result"))
