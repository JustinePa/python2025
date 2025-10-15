# Python for biologists - Autumn 2025 - Justine Pagnier

# Import libraries
import sys                      #for argv
from pathlib import Path        # to build paths
import pandas as pd             #to handle dataframes
import folium                   #to build the map
from statistics import mean     #little computation needed to center the map in the right place

# Arguments
# Usage: python map_species.py "<species>" input_csv

# ----------------------------
# Simple start menu (interactive)
# ----------------------------
DO_SINGLE = True
DO_ALL    = False
DO_CHECK  = False 

# If args exist, keep working as before; otherwise prompt interactively
if len(sys.argv) >= 3:
    species = sys.argv[1]
    in_path = Path(sys.argv[2])
else:
    print("\nChoose a mode:")
    print("  1) Plot ONE species")
    print("  2) Plot ALL species (layered)")
    print("  3) Plot ONE species and CHECK a NEW detection point\n")
    choice = input("Enter 1 / 2 / 3: ").strip()

    csv_path = input("Path to input CSV (e.g., inputs/occurrences.csv): ").strip()
    in_path = Path(csv_path)

    if choice == "2":
        DO_SINGLE = False
        DO_ALL    = True
        species   = "(all)"   # placeholder; not used in 'all' mode
    elif choice == "3":
        DO_SINGLE = False
        DO_ALL    = False
        DO_CHECK  = True
        species   = input('Species name for plotting/check (e.g., "Ostreopsis ovata"): ').strip()
        # collect the new detection info now; we'll use it later
        try:
            NEW_LAT  = float(input("New detection Latitude (e.g., 57.7): ").strip())
            NEW_LON  = float(input("New detection Longitude (e.g., 11.9): ").strip())
            THR_STR  = input("Outlier threshold in km (default 200): ").strip()
            NEW_THR  = float(THR_STR) if THR_STR else 200.0
        except Exception as _e:
            print(f"(Invalid new detection inputs; will skip check)  Details: {_e}")
            DO_CHECK = False
    else:
        # default to mode 1 if user typed something odd
        DO_SINGLE = True
        DO_ALL    = False
        species   = input('Species name to plot (e.g., "Ostreopsis ovata"): ').strip()

# Create a path for the output map (works for all modes)
out_path = Path("outputs") / f"{species}.html"
out_path.parent.mkdir(parents=True, exist_ok=True)


# Read files

df = pd.read_csv(in_path)  # expects: species, latitude, longitude

# do a safety check to make sure all needed columns are here
missing_cols = [col for col in ("species", "latitude", "longitude") if col not in df.columns]
if missing_cols:
    sys.exit(f"CSV missing required columns: {missing_cols}")

# Clean data frame
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce") # convert to numeric, and to NaN if cannot be converted
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce") # convert to numeric, and to NaN if cannot be converted
df = df.dropna(subset=["latitude", "longitude"]) # remove rows with missing coordinates
df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))] # make sure all coordinates are in the correct extent

if DO_SINGLE:
    sp_df = df[df["species"] == species]  # unchanged

    if sp_df.empty:
        avail = ", ".join(sorted(map(str, df["species"].dropna().unique())))
        sys.exit(f'No rows for species "{species}". Available (first 10): {avail}')

    # Library folium documentation: https://python-visualization.github.io/folium/latest/user_guide.html
    # Tell the program where to center the map based on the given coordinates, how much to zoom and which tiles to use
    # Other tiles can be explore here: https://leaflet-extras.github.io/leaflet-providers/preview/ 
    # Update!! Not all tiles on this website are available (some removed their free tile server and now need an API key...)

    center_lat = mean(sp_df["latitude"])
    center_lon = mean(sp_df["longitude"])
    map = folium.Map(location=(center_lat, center_lon), zoom_start=3, tiles="OpenStreetMap.Mapnik")

    # Draw a point for each lat/lon combination
    for i in range(len(sp_df)):
        species = sp_df["species"].iloc[i]
        lat = sp_df["latitude"].iloc[i]
        lon = sp_df["longitude"].iloc[i]

        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            popup=f'{species}<br>{lat:.4f}, {lon:.4f}',
            fill=True,
            fill_opacity=0.7,
            ).add_to(map)

    map.save(str(out_path))

    # ---- outputs
    print(f'Species: "{species}"')
    print(f"Points mapped: {len(sp_df)}")
    print(f"Map saved to : {out_path}")

    # --- clustered map  ---
    from folium.plugins import MarkerCluster

    cluster_map = folium.Map(location=(center_lat, center_lon), zoom_start=3, tiles="OpenStreetMap.Mapnik")
    mc = MarkerCluster(name="Occurrences").add_to(cluster_map)

    for _, row in sp_df.iterrows():
        folium.CircleMarker(
            location=(float(row["latitude"]), float(row["longitude"])),
            radius=4,
            popup=f'{row["species"]}<br>{float(row["latitude"]):.4f}, {float(row["longitude"]):.4f}',
            fill=True,
            fill_opacity=0.7,
        ).add_to(mc)

    # fit to data bounds
    min_lat, max_lat = float(sp_df["latitude"].min()),  float(sp_df["latitude"].max())
    min_lon, max_lon = float(sp_df["longitude"].min()), float(sp_df["longitude"].max())
    cluster_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    cluster_out = out_path.with_name(out_path.stem + "_cluster.html")
    cluster_map.save(str(cluster_out))
    print(f"Clustered map saved to: {cluster_out}")

    # --- heatmap ---
    from folium.plugins import HeatMap

    heat_map = folium.Map(location=(center_lat, center_lon), zoom_start=3, tiles="OpenStreetMap.Mapnik")

    # HeatMap wants a list of [lat, lon]
    pts = sp_df[["latitude", "longitude"]].astype(float).values.tolist()
    HeatMap(pts, radius=12, blur=15, max_zoom=6).add_to(heat_map)

    # fit to data bounds
    heat_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    heat_out = out_path.with_name(out_path.stem + "_heatmap.html")
    heat_map.save(str(heat_out))
    print(f"Heatmap saved to: {heat_out}")



# ---- ALL-SPECIES MODE ----
if DO_ALL:
    from folium import FeatureGroup, LayerControl
    from folium.plugins import MarkerCluster

    # toggle: set to True if you want clustered markers
    USE_CLUSTERS = True

    # collect unique species (cleaned) and pick colors
    all_df = df.dropna(subset=["species", "latitude", "longitude"]).copy()
    all_df["species"] = all_df["species"].astype(str).str.strip()
    species_list = sorted(all_df["species"].unique())

    # color palette (cycles if > len(colors))
    COLORS = [
        "red","blue","green","purple","orange","darkred","lightred","beige","darkblue",
        "darkgreen","cadetblue","darkpurple","white","pink","lightblue","lightgreen",
        "gray","black","lightgray"
    ]

    # center + bounds for the whole dataset
    all_min_lat, all_max_lat = float(all_df["latitude"].min()),  float(all_df["latitude"].max())
    all_min_lon, all_max_lon = float(all_df["longitude"].min()), float(all_df["longitude"].max())
    all_center_lat = float(all_df["latitude"].mean())
    all_center_lon = float(all_df["longitude"].mean())

    all_map = folium.Map(location=(all_center_lat, all_center_lon), zoom_start=2, tiles="OpenStreetMap.Mapnik")

    # add one layer per species
    for idx, sp in enumerate(species_list):
        color = COLORS[idx % len(COLORS)]
        layer = FeatureGroup(name=sp, show=(idx < 8))  # show first few layers by default

        if USE_CLUSTERS:
            group = MarkerCluster(name=sp).add_to(layer)
            target = group
        else:
            target = layer

        sub = all_df[all_df["species"] == sp]
        for _, row in sub.iterrows():
            lat = float(row["latitude"]); lon = float(row["longitude"])
            folium.CircleMarker(
                location=(lat, lon),
                radius=4,
                popup=f'{sp}<br>{lat:.4f}, {lon:.4f}',
                fill=True,
                fill_opacity=0.75,
                color=color,
                fill_color=color,
            ).add_to(target)

        layer.add_to(all_map)

    #  fit to bounds and add layer control
    all_map.fit_bounds([[all_min_lat, all_min_lon], [all_max_lat, all_max_lon]])
    LayerControl(collapsed=False).add_to(all_map)

    # save
    all_out = out_path.with_name("all_species_layers.html")
    all_map.save(str(all_out))
    print(f"All-species layered map saved to: {all_out}")


# ---- CHECK MODE ----
import math

# Calculate the straight-line distance (in km) between two locations, accounting for the planet’s curvature.
def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088 # mean radius of Earth in kilometers
    lat_r_1 = math.radians(lat1) # we need the coordinates in radians
    lon_r_1 = math.radians(lon1)
    lat_r_2 = math.radians(lat2)
    lon_r_2 = math.radians(lon2)
    # differences in radians between the two locations.
    da = lat_r_2 - lat_r_1 
    do = lon_r_2 - lon_r_1
    # calculate the central angle between the two points and convert it to km
    a = math.sin(da/2)**2 + math.cos(lat_r_1)*math.cos(lat_r_2)*math.sin(do/2)**2
    return 2 * radius * math.asin(math.sqrt(a))

# build convex hull around the points (https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain)
# aka. the smallest convex polygon that contains all the points
# 1. Sort all points by longitude (x) and then by latitude (y). → This orders occurrences from left to right on the map.
# 2. Build two “chains”: A lower hull (tracing the southern boundary) + An upper hull (tracing the northern boundary)
# Each chain is built by adding points one by one and checking whether each new point makes a “right turn” or “left turn.”
# If a point makes the shape bend inward (non-convex), it’s removed.
# This “cross product” test keeps only the outermost points.
# When both chains are complete, they’re joined to form the full convex polygon.
def convex_hull_latlon(points):
    # points: list of (lat, lon)
    P = sorted(points, key=lambda x: (x[1], x[0]))  # sort by lon, then lat
    if len(P) <= 1:
        return P
    def cross(o, a, b):
        # cross product (o->a) x (o->b) in lon/lat plane
        return (a[1]-o[1])*(b[0]-o[0]) - (a[0]-o[0])*(b[1]-o[1])
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

# Ray casting point-in-polygon for lat/lon tuples (https://en.wikipedia.org/wiki/Point_in_polygon)
def point_in_polygon(point, polygon):
    # point: (lat, lon); polygon: list[(lat, lon)]
    x, y = point[1], point[0]  # use lon=x, lat=y for 2D test
    inside = False # initialize
    n = len(polygon)
    if n < 3: # safety check
        return False
    for i in range(n):
        x1, y1 = polygon[i][1], polygon[i][0]
        x2, y2 = polygon[(i + 1) % n][1], polygon[(i + 1) % n][0]
        # Check if edge intersects horizontal ray
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
    return inside

def evaluate_detection(species_name, lat, lon, nn_outlier_km=200.0):
    # Pick the species subset (exact match on your cleaned df)
    subset = df[df["species"] == species_name].copy()
    if subset.empty:
        print(f' !! No rows for species "{species_name}" in the CSV.')
        return

    # Build points list
    pts = [(float(r["latitude"]), float(r["longitude"])) for _, r in subset.iterrows()]

    # Convex hull
    hull = convex_hull_latlon(pts) if len(pts) >= 3 else pts[:]  # need >=3 for a polygon
    in_hull = point_in_polygon((lat, lon), hull) if len(hull) >= 3 else False

    # Nearest-neighbour distance to existing points
    dists = [haversine_km(lat, lon, p[0], p[1]) for p in pts]
    nn_km = min(dists) if dists else float("inf")
    nn_flag = nn_km > nn_outlier_km

    # Verdict text
    verdict_bits = []
    verdict_bits.append("The new occurrence is INSIDE convex-hull range" if in_hull else "The new occurrence is OUTSIDE convex-hull range")
    verdict_bits.append(f"Nearest known point is {nn_km:.1f} km away")
    verdict_bits.append("Clear range shift here!" if nn_flag else "Considered in previously known distribution range")
    verdict = " | ".join(verdict_bits)
    print("VERDICT:", verdict)

    # Save a small HTML showing hull (if available), known pts, and the new detection
    try:
        local_map = folium.Map(location=(lat, lon), zoom_start=3, tiles="OpenStreetMap.Mapnik")
        # known points
        for p in pts:
            folium.CircleMarker(location=p, radius=3, fill=True, fill_opacity=0.7).add_to(local_map)
        # hull polygon
        if len(hull) >= 3:
            folium.PolyLine(locations=hull + [hull[0]], weight=2, opacity=0.8).add_to(local_map)
        # new detection (highlight)
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            fill=True,
            fill_opacity=0.9,
            color="red",
            fill_color="red",
            popup=f"NEW DETECTION<br>{species_name}<br>{lat:.4f}, {lon:.4f}<br>NN={nn_km:.1f} km"
        ).add_to(local_map)

        local_map.fit_bounds([
            [min([p[0] for p in pts] + [lat]), min([p[1] for p in pts] + [lon])],
            [max([p[0] for p in pts] + [lat]), max([p[1] for p in pts] + [lon])]
        ])

        out_html = out_path.with_name(f"{species_name}_new_detection_check.html")
        local_map.save(str(out_html))
        print(f"Map with hull & new point: {out_html}")
    except Exception as e:
        print(f"(Could not create local map for the detection: {e})")

    # Write a small log or txt file for the result recap
    res_txt = out_path.with_name(f"{species_name}_new_detection_result.txt")
    with open(res_txt, "w", encoding="utf-8") as f:
        f.write(f"Species: {species_name}\n")
        f.write(f"New detection: {lat:.6f}, {lon:.6f}\n")
        f.write(f"Inside convex hull? {in_hull}\n")
        f.write(f"Nearest known point (km): {nn_km:.2f}\n")
        f.write(f"Outlier threshold (km): {nn_outlier_km:.1f}\n")
        f.write(f"NN farther than threshold? {nn_flag}\n")
    print(f"Result saved to: {res_txt}")

# ---- NEW DETECTION CHECK ----
if DO_CHECK:
    evaluate_detection(species_name=species, lat=NEW_LAT, lon=NEW_LON, nn_outlier_km=NEW_THR)


