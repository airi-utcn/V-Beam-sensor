import streamlit as st

AVAILABLE_MODELS = ["MLP", "MLP_FILTER", "PINN", "PINN_FILTER"]
DEFAULT_INDEX    = AVAILABLE_MODELS.index("MLP_FILTER")

_DESCRIPTIONS = {
    "MLP":        "Multi-Layer Perceptron - full dataset",
    "MLP_FILTER": "Multi-Layer Perceptron - filtered dataset",
    "PINN":       "Physics-Informed Neural Network — full dataset",
    "PINN_FILTER":"Physics-Informed Neural Network — filtered dataset",
}

#PUBLIC
def render() -> str:
    st.subheader("Model")

    selected = st.selectbox(
        label="Forward model",
        options=AVAILABLE_MODELS,
        index=DEFAULT_INDEX,
        label_visibility="collapsed",
    )

    st.caption(_DESCRIPTIONS[selected])
    return selected
