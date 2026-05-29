import streamlit as st

DISP_MIN          = 0.025
DISP_MAX_FULL     = 0.300
DISP_MAX_FILTERED = 0.150
DISP_DEFAULT      = 0.100
DISP_STEP         = 0.001

DELTA_T_MIN       = 20.0
DELTA_T_MAX       = 70.0
DELTA_T_DEFAULT   = 45.0
DELTA_T_STEP      = 0.5

_FILTERED_MODELS  = {"MLP_FILTER", "PINN_FILTER"}


def _init_state():
    defaults = {
        "disp":         DISP_DEFAULT,
        "_disp_input":  DISP_DEFAULT,
        "delta_t":      DELTA_T_DEFAULT,
        "_delta_t_input": DELTA_T_DEFAULT,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _synced_row(label, state_key, input_key, min_val, max_val, step, fmt, unit):
    st.markdown(f"**{label}**")

    def _from_slider():
        st.session_state[input_key] = st.session_state[state_key]

    def _from_input():
        val = st.session_state[input_key]
        st.session_state[state_key] = max(min_val, min(max_val, val))

    st.slider(
        label            = label,
        min_value        = min_val,
        max_value        = max_val,
        step             = step,
        format           = f"%{fmt}",
        key              = state_key,
        on_change        = _from_slider,
        label_visibility = "collapsed",
    )

    st.number_input(
        label     = f"Enter value ({unit})",
        min_value = min_val,
        max_value = max_val,
        step      = step,
        format    = f"%{fmt}",
        key       = input_key,
        on_change = _from_input,
    )

#PUBLIC
def render(model_type: str = "MLP_FILTER") -> tuple:
    _init_state()

    disp_max = DISP_MAX_FILTERED if model_type in _FILTERED_MODELS else DISP_MAX_FULL

    if st.session_state["disp"] > disp_max:
        st.session_state["disp"]        = disp_max
        st.session_state["_disp_input"] = disp_max

    st.subheader("Parameters")

    _synced_row(
        label     = f"Target Displacement  [{DISP_MIN} – {disp_max}] mm",
        state_key = "disp",
        input_key = "_disp_input",
        min_val   = DISP_MIN,
        max_val   = disp_max,
        step      = DISP_STEP,
        fmt       = ".3f",
        unit      = "mm",
    )

    st.markdown("")

    _synced_row(
        label     = f"Delta Temperature  [{int(DELTA_T_MIN)} – {int(DELTA_T_MAX)}] °C",
        state_key = "delta_t",
        input_key = "_delta_t_input",
        min_val   = DELTA_T_MIN,
        max_val   = DELTA_T_MAX,
        step      = DELTA_T_STEP,
        fmt       = ".1f",
        unit      = "°C",
    )

    return st.session_state["disp"], st.session_state["delta_t"]
