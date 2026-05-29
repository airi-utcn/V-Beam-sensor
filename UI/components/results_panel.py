import streamlit as st
from typing import Optional

#PUBLIC
def render(result: Optional[dict]):
    st.subheader("Results")

    if result is None:
        st.info("Set the parameters in the sidebar and press **PREDICT** to run.")
        return

    st.markdown("##### Geometry")
    c1, c2, c3 = st.columns(3)
    c1.metric("Beta Angle",   f"{result['beta']:.3f} °")
    c2.metric("Beam Length",  f"{result['beam_length']:.3f} mm")
    c3.metric("Beam Width",   f"{result['beam_width']:.3f} mm")

    st.divider()

    st.markdown("##### Predictions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Displacement", f"{result['pred_displacement']:.4f} mm")
    c2.metric("Stress",       f"{result['pred_stress']:.2f} MPa")
    c3.metric("Volume",       f"{result['pred_vol']:.2f} mm³")
