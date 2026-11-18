"""
PathSnap Evaluator Dashboard (Laptop 2)
Author: PathSnap Team
"""

import streamlit as st
import requests
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Header
st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption("Distributed Evaluation Interface running on Laptop 2 | Communicating with Model & HMM API on Laptop 1")

# Sidebar Configuration
st.sidebar.header("⚙️ Server Configuration")
api_base = st.sidebar.text_input(
    "Laptop 1 API Base URL",
    value="http://localhost:8000",
    help="Enter the IP address of Laptop 1 (e.g., http://192.168.1.15:8000)"
).rstrip("/")

# Health Check & Fetch Trips
trips = []
partition_name = "evaluator_test"
try:
    resp = requests.get(f"{api_base}/trips", timeout=3)
    if resp.status_code == 200:
        data = resp.json()
        trips = data.get("trips", [])
        partition_name = data.get("partition", "evaluator_test")
        st.sidebar.success(f"🟢 Connected to Laptop 1 API ({len(trips)} {partition_name} trips available)")
    else:
        st.sidebar.warning(f"⚠️ API returned status {resp.status_code}")
except Exception as e:
    st.sidebar.error(f"🔴 Cannot reach Laptop 1 API at {api_base}")
    # Fallback to known evaluation trips if disconnected
    trips = [
        "00000015", "00000017", "00000023", "00000030", "00000049",
        "00000059", "00000067", "00000073", "00000087", "00000094"
    ]

# Evaluation Selection
st.sidebar.subheader("🎯 Evaluation Selection")
selected_trip = st.sidebar.selectbox(
    f"Select Reserved Trip ({partition_name})",
    options=trips if trips else ["00000015"],
    index=0
)

predict_btn = st.sidebar.button("🚀 Reconstruct Route & Evaluate", type="primary", use_container_width=True)

# Main Application Window
if predict_btn or ("last_trip" in st.session_state and st.session_state["last_trip"] == selected_trip):
    st.session_state["last_trip"] = selected_trip

    with st.spinner(f"Requesting inference for Trip {selected_trip} from Laptop 1..."):
        try:
            req_url = f"{api_base}/predict/{selected_trip}"
            res = requests.get(req_url, timeout=30)

            if res.status_code == 200:
                result = res.json()
                metrics = result.get("metrics", {})
                timing = result.get("timing", {})

                # Metric Cards
                st.subheader(f"📊 Evaluation Results for Trip `{selected_trip}`")
                c1, c2, c3, c4 = st.columns(4)

                top1_acc = metrics.get("accuracy", 0.0)
                continuity = metrics.get("continuity", 0.0)
                matched = metrics.get("matched", 0)
                total = metrics.get("total", 0)
                total_time = timing.get("total_s", 0.0)

                c1.metric("Top-1 Observation Accuracy", f"{top1_acc * 100:.1f}%")
                c2.metric("Route Continuity", f"{continuity * 100:.1f}%")
                c3.metric("Matched Arcs", f"{matched} / {total}")
                c4.metric("Inference Latency", f"{total_time:.2f} s")

                # Map Display
                st.markdown("---")
                st.subheader("🗺️ Interactive Trajectory Reconstruction")
                st.caption(
                    "🔵 **Blue circles**: Raw GPS observations with urban multipath noise | "
                    "🔴 **Red line**: PathSnap ML + HMM reconstructed route | "
                    "🟢 **Green line**: Ground truth road segment traversal"
                )

                map_html = result.get("map_html", "")
                if map_html:
                    components.html(map_html, height=620, scrolling=True)
                else:
                    st.warning("No map rendering received from server.")

            else:
                st.error(f"Server Error {res.status_code}: {res.text}")

        except Exception as e:
            st.error(f"Failed to communicate with Laptop 1 API: {e}")

else:
    # Landing instructions
    st.info("👈 **Select any of the 10 reserved evaluation trips in the sidebar and click 'Reconstruct Route & Evaluate'.**")
    st.markdown("""
    ### System Architecture:
    - **Laptop 1 (Backend)**: Runs the FastAPI server hosting the trained **MLP Neural Network Classifier** + **Viterbi HMM Decoder**.
    - **Laptop 2 (Frontend)**: Runs this Streamlit evaluation dashboard.
    - **Security & Integrity**: Ground-truth route labels remain strictly server-side on Laptop 1 and are never transmitted to the client.
    """)
