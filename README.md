# Python for Biologists – Autumn 2025
**Author:** Justine Pagnier

Interactive mapping of marine species occurrences using **Folium** (Leaflet).  
The script reads a CSV of occurrences (`species, latitude, longitude`) and produces HTML maps you can open in any browser.  
It can:
1. Plot a **single species** (with normal, clustered, and heatmap views)  
2. Plot **all species at once**, each with a unique color and switchable layers  
3. **Check a new detection** to see if it falls inside or outside the known range  

---

## Directory structure
- `map_species.py` – Python script to generate species maps  
- `inputs/occurrences.csv` – Example dataset of occurrence records
  
```
your-project/
├─ map_species.py
├─ README.md
├─ requirements.txt
├─ inputs/
│ └─ occurrences.csv # small test file (provided below)
└─ outputs/ # created automatically for results
```
---

## Input data
The file `occurrences.csv` must contain **3 columns**:

| species | latitude | longitude |
|---------|----------|-----------|

**Notes:**
- Latitude and longitude are in **decimal degrees** (WGS84).  
- Include a **header row** exactly as shown above.  
- You can use your own CSV as long as it follows the same structure.

---

## Usage

Install the requirements in a virtual environement (recommended): 
```bash
pip install -r requirements.txt
```
Run the script from the command line and use the interactive prompts:

```bash
python map_species.py
```
