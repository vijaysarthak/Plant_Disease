"""
streamlit_app.py — the FRONTEND: a web page where a farmer uploads a leaf photo and gets advice.

Run it from the project folder:
    streamlit run app/streamlit_app.py
The browser opens http://localhost:8501

How it is connected:
    this page  ->  src/predictor.py  ->  saved_model/ (files from trained_model.zip, made on Colab)
               ->  src/database.py   ->  database/predictions.db (history + feedback)

Streamlit re-runs this whole file from top to bottom every time the user clicks something.
That is why we remember results in `st.session_state` (a dictionary that survives the re-runs).
"""
import sys
from pathlib import Path

# Make the project folder importable (so "import config" and "from src ... import" work)
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import streamlit as st
from PIL import Image

from src import database
from src.disease_info import CLASS_NAMES, DISEASES, GROUP_CODES, GROUP_NAMES, pretty_class_name
from src.predictor import PlantDiseasePredictor

# ----------------------------------------------------------------------------
# Page settings and a little styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Plant Disease Advisor", page_icon="🌿", layout="wide")

st.markdown("""
<style>
/* See-through colours: the text keeps the theme's own colour, so the card is readable in dark AND light mode */
.result-card {padding: 1.1rem 1.3rem; border-radius: 12px; margin-bottom: 1rem;
              border: 1px solid; border-left: 8px solid; color: inherit;}
.result-card h3 {margin: 0 0 .4rem 0; color: inherit;}
.result-card p  {margin: 0; color: inherit; opacity: 0.9;}
.advise  {background: rgba(46, 204, 113, 0.15); border-color: #2ecc71;}
.refer   {background: rgba(243, 156, 18, 0.15); border-color: #f39c12;}
.unknown {background: rgba(149, 165, 166, 0.18); border-color: #95a5a6;}
.chip {display: inline-block; padding: 2px 10px; margin: 2px; border-radius: 12px;
       background: #e8eaf6; color: #283593; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load the model ONCE (st.cache_resource keeps it in memory between re-runs)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return PlantDiseasePredictor()


try:
    predictor = get_predictor()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Train the model with train_model_colab.ipynb on Google Colab, unzip trained_model.zip "
            "into the saved_model folder, then refresh this page.")
    st.stop()

database.create_tables()

# ----------------------------------------------------------------------------
# Sidebar: project name, team and supported crops
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🌿 Plant Disease Advisor")
    st.markdown("**Created by**  \nSarthak Vijay, Himanshu Kumar Singh, Arun Das, Sahil Kumar")
    st.divider()
    st.subheader("Supported crops")
    crops = sorted(set(name.split("___")[0].replace("_", " ").replace(",", "") for name in CLASS_NAMES))
    st.write(", ".join(crops))

# ----------------------------------------------------------------------------
# Three tabs
# ----------------------------------------------------------------------------
tab_diagnose, tab_history, tab_how = st.tabs(["🔍 Diagnose a leaf", "🕘 History", "📖 How it works"])

# =============================== TAB 1: DIAGNOSE ===============================
with tab_diagnose:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.subheader("1. Upload a photo of ONE leaf")
        source = st.radio("Photo source", ["Upload a file", "Use the camera"], horizontal=True,
                          label_visibility="collapsed")
        if source == "Upload a file":
            photo_file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])
        else:
            photo_file = st.camera_input("Take a photo")
        st.caption("Tip: daylight, sharp focus, the leaf filling most of the photo.")

        if photo_file is not None:
            image = Image.open(photo_file).convert("RGB")
            st.image(image, width="stretch")

    with right:
        st.subheader("2. Result")
        if photo_file is None:
            st.info("Upload a photo on the left to get a diagnosis.")
        else:
            # Predict only when a NEW photo arrives (not on every click)
            photo_id = f"{photo_file.name}-{photo_file.size}"
            if st.session_state.get("photo_id") != photo_id:
                result = predictor.predict(image)
                st.session_state["photo_id"] = photo_id
                st.session_state["result"] = result
                st.session_state["prediction_id"] = database.save_prediction(photo_file.name, result)
                st.session_state["feedback_sent"] = False
            result = st.session_state["result"]
            advice = result["advice"]

            # ---- the decision card ----
            if result["decision"] == "advise":
                heading = f"✅ Treatment group: {result['advised_group']}"
            elif result["decision"] == "refer":
                heading = "⚠️ Not sure — please ask an expert"
            else:
                heading = "❓ Photo not recognised"
            st.markdown(f"""
            <div class="result-card {result['decision']}">
              <h3>{heading}</h3>
              <p>{advice['what_it_means']}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("**What to do**")
            st.write(advice["what_to_do"])
            st.caption(f"Source: {advice['source']}")

            # ---- details ----
            with st.expander("Why this answer? (details)", expanded=result["decision"] != "advise"):
                chips = "".join(f'<span class="chip">{name}</span>' for name in result["prediction_set"]) or "—"
                st.markdown(f"**Groups the model cannot rule out (prediction set):** {chips}",
                            unsafe_allow_html=True)
                st.caption("One group → advice. Two or more → the treatments differ, so we refer to an expert.")

                st.markdown("**Probability of each treatment group**")
                for group_name, probability in result["group_probabilities"].items():
                    st.progress(min(probability, 1.0), text=f"{group_name}: {probability:.1%}")

                st.markdown("**Most likely diseases**")
                for item in result["top_classes"]:
                    st.write(f"- {item['pretty_name']} — {item['probability']:.1%}  "
                             f"*(cause: {item['cause']}, group: {item['group']})*")
                if result["is_unfamiliar"]:
                    st.write(f"Energy score {result['energy']:.2f} is above the limit "
                             f"{predictor.energy_threshold:.2f}: the photo looks unlike the training photos.")

            # ---- feedback ----
            st.subheader("3. Was this right?")
            if st.session_state["feedback_sent"]:
                st.success("Thank you! Your feedback was saved.")
            else:
                with st.form("feedback"):
                    answer = st.radio("Answer", ["Yes", "No", "I don't know"], horizontal=True)
                    true_class = st.selectbox("If you know the real disease, choose it (optional)",
                                              ["—"] + CLASS_NAMES, format_func=lambda n: n if n == "—" else pretty_class_name(n))
                    comment = st.text_input("Comment (optional)")
                    if st.form_submit_button("Send feedback"):
                        if answer != "I don't know":
                            database.save_feedback(st.session_state["prediction_id"], answer == "Yes",
                                                   "" if true_class == "—" else true_class, comment)
                        st.session_state["feedback_sent"] = True
                        st.rerun()

# =============================== TAB 2: HISTORY ===============================
with tab_history:
    st.subheader("Last 50 predictions")
    history = database.get_history(limit=50)
    if len(history) == 0:
        st.info("No predictions yet.")
    else:
        table = pd.DataFrame(history)
        table["top_class"] = table["top_class"].map(pretty_class_name)
        table["top_probability"] = (table["top_probability"] * 100).round(1).astype(str) + " %"
        table["is_correct"] = table["is_correct"].map({1: "✅ yes", 0: "❌ no"}).fillna("")
        table = table.rename(columns={"id": "#", "created_at": "time", "image_name": "photo",
                                      "top_class": "most likely disease", "top_probability": "probability",
                                      "decision": "decision", "advised_group": "advised group",
                                      "prediction_set": "prediction set", "is_correct": "feedback"})
        st.dataframe(table, width="stretch", hide_index=True)

# =============================== TAB 3: HOW IT WORKS ===============================
with tab_how:
    st.subheader("How the advisor decides")
    st.markdown("""
1. **The network** (trained in `train_model_colab.ipynb` on Google Colab) gives a probability for each of 38 diseases.
2. The diseases are put into **6 treatment groups** — what the farmer has to *do*.
   The probability of a group = sum of its diseases' probabilities.
3. Every group has a **threshold** learned from labelled field photos (*conformal prediction*).
   All groups that pass are the **prediction set**. It contains the correct group about 90 % of the time.
4. **Decision:**
   * exactly **one** group → give that group's advice
   * **two or more** groups → the treatments would differ → **refer to an expert (KVK)**
   * the photo looks unlike anything seen in training (high *energy* score) → **photo not recognised**

This is safer than trusting the single most likely disease: on field photos a normal classifier is often
wrong while being confident.
""")
    group_table = []
    for code in GROUP_CODES:
        members = [pretty_class_name(n) for n in CLASS_NAMES if DISEASES[n][1] == code]
        group_table.append({"group": f"{code} {GROUP_NAMES[code]}", "diseases": ", ".join(members)})
    st.table(pd.DataFrame(group_table))
    st.caption("⚠️ This tool supports, but does not replace, an agriculture expert. "
               "Always follow the label of a registered (CIB&RC) product.")