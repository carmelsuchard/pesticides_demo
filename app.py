from pathlib import Path
from datetime import date, datetime
import html
import json
import re

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from branca.element import Element
from folium.plugins import Fullscreen, LocateControl
from pyproj import Transformer
from streamlit_folium import st_folium
from data.brp_options import BRP_options


st.set_page_config(
    page_title="Exposure map",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).with_name("data")
# Fixed study-area extent from bounding_box.gpkg, in EPSG:28992.
STUDY_BOUNDS_RD = (167906.232375, 435256.594, 183865.587, 446770.671)
STUDY_AREA_ASPECT_RATIO = (STUDY_BOUNDS_RD[2] - STUDY_BOUNDS_RD[0]) / (
    STUDY_BOUNDS_RD[3] - STUDY_BOUNDS_RD[1]
)
WGS84_FROM_RD = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def load_styles() -> None:
    css_path = Path(__file__).with_name("styles.css")
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def add_map_chrome(map_object: folium.Map) -> None:
    map_object.get_root().html.add_child(
        Element(
            """
            <style>
                .map-status, .map-attribution-note, .map-legend { font-family: 'Aptos', 'Segoe UI', sans-serif; }
                .map-status { align-items: center; background: rgba(255,255,255,.96); border: 1px solid #d4e4de; border-radius: 999px; color: #173c35; display: flex; font-size: 13px; gap: 7px; left: 16px; padding: 8px 12px; position: absolute; top: 16px; z-index: 1000; }
                .status-dot { background: #43a56f; border-radius: 50%; display: inline-block; height: 7px; width: 7px; }
                .map-attribution-note { bottom: 7px; color: #71837f; font-size: 10px; position: absolute; right: 10px; z-index: 1000; }
                .map-legend { background: rgba(255,255,255,.98); border: 2px solid #7f908b; border-radius: 10px; bottom: 18px; box-shadow: 0 8px 24px rgba(27,71,60,.2); color: #16352f; font-size: 12px; left: 16px; padding: 14px; position: absolute; width: 238px; z-index: 1000; }
                .legend-title { color: #16352f; font-size: 13px; font-weight: 700; margin-bottom: 10px; }
                .legend-item { align-items: center; display: flex; gap: 10px; line-height: 1.3; }
                .legend-swatch { background: #aeb9b6 !important; border: 3px solid #7f908b !important; border-radius: 5px; display: block; flex: 0 0 auto; height: 28px !important; width: 28px !important; }
                .far-house-swatch { background: #aeb9b6 !important; border-color: #7f908b !important; }
                .legend-emphasis { display: block; font-weight: 700; }
                @media (max-width: 700px) { .map-legend { bottom: 10px; left: 10px; width: 180px; } }
            </style>
            <div class="map-attribution-note">Base map · OpenStreetMap</div>
            <div class="map-legend">
                <div class="legend-item">
                    <span class="legend-swatch far-house-swatch"></span>
                    <span>House is farther than 250 meters from a field</span>
                </div>
            </div>
            """
        )
    )


@st.cache_data(show_spinner="Preparing map data...")
def load_layer_geojson(filename: str) -> tuple[str, list[str]]:
    layer = gpd.read_file(DATA_DIR / filename).to_crs("EPSG:4326")
    return layer.to_json(), list(layer.columns)


@st.cache_data(show_spinner="Loading weather data...")
def load_weather_data() -> pd.DataFrame:
    weather = pd.read_csv(DATA_DIR / "meteo_data_filled.csv", parse_dates=["datetime"])
    return weather.set_index("datetime")


def weather_for_selection(selected_date: date, selected_hour: int) -> dict[str, str]:
    selected_datetime = datetime.combine(selected_date, datetime.min.time()).replace(hour=selected_hour)
    weather = load_weather_data()
    if selected_datetime not in weather.index:
        return {
            "wind_speed": "--",
            "wind_direction": "--",
            "temperature": "--",
            "humidity": "--",
            "cloud_coverage": "--",
        }

    row = weather.loc[selected_datetime]
    return {
        "wind_speed": f"{float(row['wind_speed_m_s']):.1f}",
        "wind_direction": f"{row['wind_direction_letter']}",
        "temperature": f"{float(row['temperature_k']) - 273.15:.1f}",
        "humidity": f"{float(row['relative_humidity']) * 100:.0f}",
        "cloud_coverage": f"{float(row['cloud_coverage']):.0f}",
    }


def wind_bearing(direction: str) -> int:
    return {
        "N": 0,
        "NE": 45,
        "E": 90,
        "SE": 135,
        "S": 180,
        "SW": 225,
        "W": 270,
        "NW": 315,
    }.get(direction.upper(), 0)


def rd_bounds_as_lat_lon() -> list[list[float]]:
    min_x, min_y, max_x, max_y = STUDY_BOUNDS_RD
    min_lon, min_lat = WGS84_FROM_RD.transform(min_x, min_y)
    max_lon, max_lat = WGS84_FROM_RD.transform(max_x, max_y)
    return [[min_lat, min_lon], [max_lat, max_lon]]


def feature_popup(feature: dict, label: str) -> str:
    properties = feature.get("properties", {})
    identifier = html.escape(str(properties.get("identificatie", "")))
    return f"<b>{label}</b><br>identificatie: {identifier}"


def bag_id_from_popup(popup: str | dict | None) -> str | None:
    if popup is None:
        return None

    if isinstance(popup, dict):
        properties = popup.get("properties", popup)
        identifier = properties.get("identificatie") or properties.get("BAG ID")
        if identifier is not None and str(identifier).strip():
            return str(identifier).strip()

    # Flatten popup HTML into text, then look for an ID after its label.
    text = html.unescape(str(popup))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)

    match = re.search(
        r"(?:identificatie|BAG\s*ID)\s*:?\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def build_data_layers(exposure_map: folium.Map) -> None:
    try:
        bag_json, bag_columns = load_layer_geojson("BAG_clipped_simplified_2.gpkg")
        brp_json, brp_columns = load_layer_geojson("BRP_clipped_simplified_2.gpkg")
        missing_columns = {"within_250", "identificatie"}.difference(bag_columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"BAG_clipped_simplified.gpkg is missing required column(s): {missing}")
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Map data could not be loaded: {error}")
        return

    bag_features = json.loads(bag_json)
    brp_features = json.loads(brp_json)

    def bag_style(feature: dict) -> dict:
        within_250 = feature.get("properties", {}).get("within_250", False)
        is_nearby = within_250 is True or str(within_250).lower() == "true"
        return {
            "color": "#c2813a" if is_nearby else "#9aa8a5",
            "fillColor": "#e5b66c" if is_nearby else "#aeb9b6",
            "fillOpacity": 0.62 if is_nearby else 0.3,
            "weight": 1.4 if is_nearby else 1,
            "interactive": is_nearby,
        }

    def bag_highlight(feature: dict) -> dict:
        within_250 = feature.get("properties", {}).get("within_250", False)
        is_nearby = within_250 is True or str(within_250).lower() == "true"
        return {"weight": 3, "fillOpacity": 0.82 if is_nearby else 0.3}

    def bag_popup(feature: dict) -> str:
        within_250 = feature.get("properties", {}).get("within_250", False)
        if not (within_250 is True or str(within_250).lower() == "true"):
            return "<b>House outside 250 m</b><br>Not selectable"
        return feature_popup(feature, "BAG house")

    folium.GeoJson(
        bag_features,
        name="BAG houses",
        style_function=bag_style,
        highlight_function=bag_highlight,
        tooltip=folium.GeoJsonTooltip(
            fields=["identificatie"],
            aliases=["BAG ID"],
            sticky=False,
        ),
        popup=folium.GeoJsonPopup(fields=["identificatie"], aliases=["BAG ID"]),
        show=True,
    ).add_to(exposure_map)

    folium.GeoJson(
        brp_features,
        name="BRP fields",
        style_function=lambda feature: {
            "color": "#239c82",
            "fillColor": "#65b893",
            "fillOpacity": 0.28,
            "weight": 1,
        },
        highlight_function=lambda feature: {"weight": 2, "fillOpacity": 0.48},
        tooltip=folium.GeoJsonTooltip(fields=brp_columns[:3], sticky=False) if len(brp_columns) > 1 else None,
        show=True,
    ).add_to(exposure_map)


@st.cache_data(show_spinner=False)
def load_exposure_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
    
def calculate_exposure(
    bag_id: str,
    selected_date: date | datetime,
    selected_hour: int,
    active_component: str,
    crop: list[str] | tuple[str, ...] | str,
    method: list[str] | tuple[str, ...] | str,
) -> dict[str, object]:
    folder = Path(__file__).with_name("local_run_demo_october_20")

    def result(status: str, message: str, **extra) -> dict[str, object]:
        return {
            "status": status,
            "message": message,
            "bag_id": bag_id,
            "component": active_component,
            **extra,
        }

    def as_selection_list(
        selection: list[str] | tuple[str, ...] | str | None,
    ) -> list[str]:
        if selection is None:
            return []
        if isinstance(selection, str):
            return [selection.strip()]
        return [
            str(value).strip()
            for value in selection
            if str(value).strip()
        ]

    if not bag_id or not bag_id.strip():
        return result("invalid_bag_id", "Choose a house on the map.")

    if not 0 <= selected_hour <= 23:
        return result(
            "invalid_hour",
            "The selected hour must be between 0 and 23.",
        )

    # Convert datetime to date if a datetime is passed.
    if isinstance(selected_date, datetime):
        selected_date = selected_date.date()

    selected_crops = as_selection_list(crop)
    selected_methods = as_selection_list(method)

    all_crops_selected = (
        not selected_crops
        or any(value.casefold() == "all crops" for value in selected_crops)
    )
    all_methods_selected = (
        not selected_methods
        or any(value.casefold() == "all methods" for value in selected_methods)
    )

    crop_ids: set[str] = set()

    if not all_crops_selected:
        for crop_name in selected_crops:
            crop_match = re.search(r"\((\d+)\)\s*$", crop_name)
            if not crop_match:
                return result(
                    "invalid_crop",
                    f"Could not find a crop ID in {crop_name!r}.",
                )
            crop_ids.add(crop_match.group(1))

    selected_method_names = {
        value.casefold()
        for value in selected_methods
        if value
    }


    # Make it so I can search in 6 possible folders where the file may be located. This is a temporary solution until the file structure is finalized.
    search_folders = [
        Path(__file__).with_name("local_run_demo_october_20_part_1"),
        Path(__file__).with_name("local_run_demo_october_20_part_2"),
        Path(__file__).with_name("local_run_demo_october_20_part_3"),
        Path(__file__).with_name("local_run_demo_october_20_part_4"),
        Path(__file__).with_name("local_run_demo_october_20_part_5"),
        Path(__file__).with_name("local_run_demo_october_20_part_6"),
        Path(__file__).with_name("local_run_demo_october_20_part_7"),
        Path(__file__).with_name("local_run_demo_october_20_part_8"),
    ]

    pattern = f"{bag_id.strip()}_demo_run*"
    matches = sorted(
        (
            path
            for search_folder in search_folders
            for path in search_folder.glob(pattern)
            if path.is_file()
        ),
        key=str,
    )

    if not matches:
        return result(
            "file_not_found",
            "No events within application windows were found.",
        )

    if len(matches) > 1:
        return result(
            "ambiguous_file",
            "More than one file matched this BAG ID.",
            matching_files=[str(path) for path in matches],
        )

    try:
        # Copy the cached DataFrame before normalizing column names.
        # This avoids mutating the object stored by st.cache_data.
        df = load_exposure_csv(str(matches[0])).copy()
    except Exception as exc:
        return result(
            "file_read_error",
            f"Could not read {matches[0].name}: {exc}",
        )

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    component_column = active_component.strip().lower()

    if component_column not in df.columns:
        return result(
            "component_not_available",
            f"{active_component} was not found for this house.",
        )

    required_columns = ["app_event_start_datetime"]

    if not all_crops_selected:
        required_columns.append("product_use_scenario_id")

    if not all_methods_selected:
        required_columns.append("application_methods")

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return result(
            "missing_data_columns",
            "Required column(s) missing from the file: "
            + ", ".join(missing_columns),
        )

    timestamps = pd.to_datetime(
        df["app_event_start_datetime"],
        errors="coerce",
    )

    mask = timestamps.notna()

    if not all_crops_selected:
        row_crop_ids = (
            df["product_use_scenario_id"]
            .astype("string")
            .str.extract(r"(\d+)\s*$", expand=False)
        )

        mask &= row_crop_ids.isin(crop_ids).fillna(False)

    if not all_methods_selected:
        row_methods = (
            df["application_methods"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        mask &= row_methods.isin(selected_method_names).fillna(False)

    values = pd.to_numeric(
        df[component_column],
        errors="coerce",
    )

    matching_values = mask & values.notna()

    same_day = timestamps.dt.date.eq(selected_date)
    same_hour = timestamps.dt.hour.eq(selected_hour)
    same_month = (
        timestamps.dt.year.eq(selected_date.year)
        & timestamps.dt.month.eq(selected_date.month)
    )
    same_year = timestamps.dt.year.eq(selected_date.year)

    periods = {
        "selected_datetime": matching_values & same_day & same_hour,
        "selected_date": matching_values & same_day,
        "selected_month": matching_values & same_month,
        "selected_year": matching_values & same_year,
    }

    averages: dict[str, float | None] = {}
    counts: dict[str, int] = {}

    for period_name, period_mask in periods.items():
        period_values = values.loc[period_mask]

        counts[period_name] = int(period_values.count())
        averages[period_name] = (
            float(period_values.mean())
            if not period_values.empty
            else None
        )

    if not matching_values.any():
        return result(
            "no_matching_rows",
            "No numeric exposure values matched the selected filters.",
            averages=averages,
            counts=counts,
            crops=selected_crops,
            methods=selected_methods,
        )

    return result(
        "ok",
        "Exposure averages calculated.",
        averages=averages,
        counts=counts,
        selected_date=selected_date.isoformat(),
        selected_hour=selected_hour,
        crops=selected_crops,
        methods=selected_methods,
    )

# @st.cache_resource(show_spinner="Building map...")
def build_static_map() -> folium.Map:
    study_bounds = rd_bounds_as_lat_lon()
    (min_lat, min_lon), (max_lat, max_lon) = study_bounds
    exposure_map = folium.Map(
        location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
        zoom_start=13,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        min_zoom=10,
        max_zoom=18,
        max_bounds=True,
        tiles=None,
        zoom_control=False,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri',
        overlay=False,
    ).add_to(exposure_map)

    Fullscreen(position="topright", title="Open fullscreen", title_cancel="Close fullscreen").add_to(
        exposure_map
    )

    build_data_layers(exposure_map)
    exposure_map.fit_bounds(rd_bounds_as_lat_lon())

    return exposure_map


def render_weather_panel(weather: dict[str, str]) -> None:
    bearing = wind_bearing(weather["wind_direction"])
    direction_arrows = "".join(
        f'<span class="direction-arrow" style="transform: rotate({bearing}deg);">&#8593;</span>'
        for _ in range(3)
    )
    st.markdown(
        f"""
        <div class="weather-panel">
            <div class="weather-heading">Conditions at nearest weather station</div>
            <div class="weather-grid">
                <div class="weather-card"><span class="weather-icon">↝</span><span class="weather-label">Wind speed</span><strong>{weather['wind_speed']} m/s</strong></div>
                <div class="weather-card direction-card"><span class="weather-icon">◌</span><span class="weather-label">Wind direction</span><div class="direction-arrows">{direction_arrows}</div><strong>{weather['wind_direction']}</strong></div>
                <div class="weather-card"><span class="weather-icon">°</span><span class="weather-label">Temperature</span><strong>{weather['temperature']} °C</strong></div>
                <div class="weather-card"><span class="weather-icon">◒</span><span class="weather-label">Humidity</span><strong>{weather['humidity']}%</strong></div>
                <div class="weather-card"><span class="weather-icon">☁</span><span class="weather-label">Cloud cover</span><strong>{weather['cloud_coverage']}/9</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_map_legend() -> None:
    st.markdown(
        """
        <div class="streamlit-map-legend">
            <div class="streamlit-legend-item">
                <span class="streamlit-legend-swatch streamlit-modeled-swatch"></span>
                <span>Modeled house</span>
            </div>
            <div class="streamlit-legend-item">
                <span class="streamlit-legend-swatch streamlit-house-swatch"></span>
                <span>House farther than 250 meters from an agricultural field</span>
            </div>
            <div class="streamlit-legend-item">
                <span class="streamlit-legend-swatch streamlit-field-swatch"></span>
                <span>Agricultural field</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-section-label'>Selected house BAG ID</div>",
            unsafe_allow_html=True,
        )

        selected_bag_id = st.session_state.get("selected_bag_id", "")
        bag_id_display = selected_bag_id or "Select a house on the map"

        st.markdown(
            f"""
            <div class="bag-id-display">
                <span class="bag-id-icon">⌂</span>
                <span>{html.escape(bag_id_display)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        exposure_result = calculate_exposure(
            bag_id=selected_bag_id,
            selected_date=st.session_state.get("selected_date", date(2017, 7, 20)),
            selected_hour=st.session_state.get("selected_hour", 12),
            active_component=st.session_state.get(
                "active_component",
                "Acetamiprid",
            ),
            crop=st.session_state.get("crop", "All crops"),
            method=st.session_state.get("method", "All methods"),
            # Include this if calculate_exposure still requires season.
        )

        status = exposure_result.get("status", "unknown")
        averages = exposure_result.get("averages", {})
        counts = exposure_result.get("counts", {})

        def format_average(key: str) -> str:
            value = averages.get(key)
            return "-" if value is None else f"{float(value):.6f}"

        def format_count(key: str) -> str:
            value = counts.get(key)
            return "-" if value is None else f"{int(value):,}"

        if status == "component_not_available":
            st.warning(
                exposure_result.get(
                    "message",
                    "This pesticide is not available for the selected house.",
                )
            )

        elif status not in {"ok", "no_matching_rows"}:
            st.info(
                exposure_result.get(
                    "message",
                    "Exposure values are not available.",
                )
            )

        st.markdown(
            """
            <div class="sidebar-section-label summary-label">
                Deterministic exposure calculation
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="selected-exposure-card">
                <div class="selected-exposure-heading">
                    Selected day and hour
                </div>
                <div class="selected-exposure-value">
                    {format_average("selected_datetime")}
                </div>
                <div class="selected-exposure-caption">
                    Mean exposure
                </div>
                <div class="selected-exposure-count">
                    {format_count("selected_datetime")}
                    <span>valid spraying events</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-section-label summary-label">
                Distribution statistics
            </div>
            """,
            unsafe_allow_html=True,
        )

        periods = [
            ("Daily", "selected_date"),
            ("Monthly", "selected_month"),
            ("Yearly", "selected_year"),
        ]

        cards_html = "".join(
            f"""
            <div class="distribution-card">
                <div class="distribution-card-heading">{label}</div>
                <div class="distribution-card-row">
                    <span>Mean</span>
                    <strong>{format_average(key)}</strong>
                </div>
                <div class="distribution-card-row count-row">
                    <span>Valid spraying events</span>
                    <strong>{format_count(key)}</strong>
                </div>
            </div>
            """
            for label, key in periods
        )

        st.markdown(
            f"""
            <div class="distribution-list">
                {cards_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='model-run-note'>Data from latest model run on 20-10-2026</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-section-label note-label">
                This demonstrator features house-level exposure information,
                which may be subject to privacy regulations. It is intended
                only for data exploration purposes by OBO2 project members.
                It is not for public release.
            </div>
            """,
            unsafe_allow_html=True,
        )

# @st.fragment
def render_selection(weather_column) -> None:
    control_columns = st.columns([0.9, 1.35, 1.35, 1.65, 1.35], gap="medium")
    with control_columns[0]:
        st.markdown("<div class='control-label'>Year</div>", unsafe_allow_html=True)
        st.selectbox("Year", [2017], label_visibility="collapsed")

    with control_columns[1]:
        st.markdown("<div class='control-label'>Active component</div>", unsafe_allow_html=True)
        active_component = st.selectbox(
            "Active component",
            ["Acetamiprid", "Asulam", "Fludioxonil", "Lambda-cyhalothrin"],
            label_visibility="collapsed",
            key="active_component"
        )
    with control_columns[2]:
        st.markdown(
            "<div class='control-label'>Crop (BRP code)</div>",
            unsafe_allow_html=True,
        )

        crop_options = ["All crops"] + [
            str(option)
            for option in BRP_options
            if str(option).casefold() != "all crops"
        ]

        crops = st.multiselect(
            "Crop",
            options=crop_options,
            default=["All crops"],
            key="crop",
            label_visibility="collapsed",
        )

    with control_columns[3]:
        st.markdown(
            "<div class='control-label'>CTGB Method</div>",
            unsafe_allow_html=True,
        )

        method_options = [
            "All methods",
            "Gewasbehandeling (Crop treatment)",
            "Over het gewas (Over the crop)",
            "Neerwaarts spuite (Downward spraying)",
        ]

        methods = st.multiselect(
            "Method",
            options=method_options,
            default=["All methods"],
            key="method",
            label_visibility="collapsed",
        )

    with control_columns[4]:
        st.markdown("<div class='control-label'>Day & hour</div>", unsafe_allow_html=True)
        selected_date = st.date_input(
            "Date",
            value=date(2017, 7, 20),
            min_value=date(2017, 1, 1),
            max_value=date(2017, 12, 31),
            key="selected_date",
            label_visibility="collapsed",
        )

    selected_hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1,
        format="%d:00",
        key="selected_hour",
        label_visibility="collapsed",
    )

    weather = weather_for_selection(selected_date, selected_hour)

    with weather_column:
        render_weather_panel(weather)
        render_map_legend()

def main() -> None:
    load_styles()

    map_column, weather_column = st.columns([5, 1], gap="medium")
    render_selection(weather_column)
    with map_column:
        map_state = st_folium(
            build_static_map(),
            use_container_width=True,
            height=calc_map_height(),
            returned_objects=["last_object_clicked", "last_object_clicked_popup"],
            key="exposure-map",
        )

    if map_state:
        clicked_bag_id = bag_id_from_popup(
            map_state.get("last_object_clicked_popup")
        )

        if clicked_bag_id and clicked_bag_id != st.session_state.get("selected_bag_id"):
            st.session_state["selected_bag_id"] = clicked_bag_id
            st.session_state["bag_id_input"] = clicked_bag_id

    render_sidebar()

def calc_map_height() -> int:
    # Match the map frame to the fixed study-area shape so fit_bounds shows no extra area.
    return round(1080 / STUDY_AREA_ASPECT_RATIO)


if __name__ == "__main__":
    main()