"""Evaluator-facing Streamlit UI for the PathSnap API."""

import os

import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="PathSnap Evaluator", layout="wide")
st.title("PathSnap: Route Reconstruction")
st.caption(
    "Select an evaluation trip to run map matching. Route scoring is performed "
    "by the API; reference route labels are not sent to this interface."
)

default_api_url = os.getenv("PATHSNAP_API_URL", "http://10.143.193.155:8000")
api_base_url = st.sidebar.text_input(
    "Backend API URL", value=default_api_url
).strip().rstrip("/")

selected_partition = st.sidebar.selectbox(
    "Dataset Partition",
    options=["evaluator_test", "student_test", "train"],
    index=0,
    format_func=lambda p: {
        "evaluator_test": "evaluator_test (10 Reserved)",
        "student_test": "student_test (10 Development)",
        "train": "train (80 Training)",
    }.get(p, p),
    help="Select which partition to load trips from. 'evaluator_test' is the primary reserved test set."
)
request_timeout_seconds = 180


@st.cache_data(ttl=5, show_spinner=False)
def fetch_evaluation_trips(base_url, partition):
    response = requests.get(f"{base_url}/trips", params={"partition": partition}, timeout=15)
    response.raise_for_status()
    payload = response.json()
    trips = payload.get("trips")
    if not isinstance(trips, list):
        raise ValueError("The API response did not contain a 'trips' list.")
    return sorted(str(trip_id) for trip_id in trips)


try:
    trip_ids = fetch_evaluation_trips(api_base_url, selected_partition)
except requests.HTTPError as exc:
    detail = ""
    try:
        detail = exc.response.json().get("detail", "")
    except (AttributeError, ValueError):
        pass
    st.error(detail or f"The PathSnap API returned an error: {exc}")
    st.stop()
except requests.RequestException as exc:
    st.error(
        f"Could not reach the PathSnap API at {api_base_url}. "
        "Confirm the backend is running and that this URL uses the backend "
        "laptop's reachable IP address."
    )
    st.code(str(exc))
    st.stop()
except (ValueError, requests.exceptions.JSONDecodeError) as exc:
    st.error(f"Could not read the evaluation trip list: {exc}")
    st.stop()

if not trip_ids:
    st.error(f"The API returned no trips for partition '{selected_partition}'.")
    st.stop()

if selected_partition == "evaluator_test":
    if len(trip_ids) == 10:
        st.success("10 reserved evaluation trips are available.")
    else:
        st.warning(
            f"The API returned {len(trip_ids)} trips; the evaluator workflow expects 10. "
            "Check that the backend is configured with the reserved evaluation partition."
        )
else:
    st.info(f"Loaded {len(trip_ids)} trips from the '{selected_partition}' partition.")

selected_trip = st.selectbox(
    f"Select Trip ({selected_partition})",
    options=trip_ids,
    key=f"trip_select_{selected_partition}",
    format_func=lambda trip_id: f"Trip {trip_id}",
)

if st.button("Run map matching and evaluate", type="primary", use_container_width=True):
    try:
        with st.spinner(f"Processing trip {selected_trip} on the API server..."):
            response = requests.get(
                f"{api_base_url}/predict/{selected_trip}",
                timeout=request_timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
        st.session_state["pathsnap_result"] = result
        st.session_state["pathsnap_result_trip"] = selected_trip
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except (ValueError, UnboundLocalError):
            pass
        st.error(f"The API could not process trip {selected_trip}. {detail or exc}")
    except requests.RequestException as exc:
        st.error(f"The API request failed: {exc}")
    except ValueError as exc:
        st.error(f"The API returned invalid JSON: {exc}")


result = st.session_state.get("pathsnap_result")
result_trip = st.session_state.get("pathsnap_result_trip")
if result and result_trip == selected_trip:
    st.divider()
    st.subheader(f"Results for trip {selected_trip}")

    metrics = result.get("metrics") or {}
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Route accuracy",
        f"{metrics['accuracy']:.1%}" if isinstance(metrics.get("accuracy"), (int, float)) else "N/A",
    )
    metric_columns[1].metric(
        "Route continuity",
        f"{metrics['continuity']:.1%}" if isinstance(metrics.get("continuity"), (int, float)) else "N/A",
    )
    metric_columns[2].metric("Matched route arcs", metrics.get("matched", "N/A"))
    metric_columns[3].metric("GPS observations", result.get("num_gps_points", "N/A"))

    timing = result.get("timing") or {}
    if timing:
        st.caption(
            "Processing time: "
            f"{timing.get('total_s', 'N/A')} s total; "
            f"{timing.get('candidate_generation_s', 'N/A')} s candidate generation; "
            f"{timing.get('viterbi_decoding_s', 'N/A')} s route decoding."
        )

    map_html = result.get("map_html")
    if map_html:
        st.subheader("GPS observations and predicted route")
        components.html(map_html, height=620, scrolling=True)
    else:
        st.info("The API response did not include a route map.")

    with st.expander("Predicted arc IDs"):
        predicted_route = result.get("predicted_route") or []
        st.write(predicted_route if predicted_route else "No route arcs were returned.")
