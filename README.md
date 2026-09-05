This is a PGA2k utility to generate .course files from USGS LiDAR and OpenStreetView OSM, based on Chad Rockey's excellent [TCG Designer Tools](https://github.com/chadrockey/TGC-Designer-Tools).

## How-To

This Python app needs the following libs: (TODO: VENV with all deps)

- numpy (math)
- scipy (math)
- pyproj (geo)
- matplotlib (viz)
- laspy (LiDAR)
- overpy (OSM)
- shapely (vector)
- PIL (image)
- tkinter (GUI)

Steps:

1. Get LAZ files from [USGS Downloader](https://apps.nationalmap.gov/downloader/). Download at least 2km^2 extent.
2. Draw your course for OSM using [OpenStreetMap](https://openstreetmap.org/). Tag everything for golf features - fairways, holes, etc.
3. Open the app & select your working folder.
4. Click "Init" to initialize the working folder. This adds some subfolders and a project file.
5. Put your LAZ files into "laz".
6. Click "Ingest LAZ" to load the LiDAR point cloud.
7. Console will output the extent coordinates. Use them to download OSM from OpenStreetMap. Move the OSM file to the working folder.
8. Click "Ingest OSM" to load the course map.
9. Click "Ingest Course" to unpack a .course file. Start from a blank that you created in TCG 2019. (TODO: add template to source)
10. Click "Generate Terrain" to do a coarse (~500 stamp) hex grid to roughly match LAZ.
11. Click "Refine Terrain" to create smaller stamps to add detail. Play with the settings for best results; rollover for details. Use the mask to refine selected areas, and adjust buffer to include or exclude areas. Select mask objects under "Splines" tab.
12. Shortcuts: mouse wheel toggles preview versions; shift-wheel changes preview selection; ctrl-wheel zooms; wheel click-hold moves map.
13. Click "Write Terrain" to create the heightmap.
14. Under "Splines", click "Write Splines" to generate course objects from OSM.
15. Under "Splines", filter by "hole" to select up to 18 holes for your course (apply Mask to exclude holes from output). Click "Write Holes" to add holes.
16. Click "Repack" to output the .course file to the working directory (give it a name first; ".course" suffix is not needed).
17. Click "Copy to Game Folder" to move the .course file to the appropriate PGA2K directory. Only v2019 is supported but .course can be opened in v2021. (TODO: compatibility)

### Funsies

1. Place "stakes" at the corners of buildings for easier placement, clear when done
2. Apply a tight landscaping on cart paths so they line up nicely
3. Import collections of objects & splines (e.g. house with landscaping, driveway, car, patio set, deck, pergola; hole marker with mulch, bench, ball washer, bin; custom combinations of fencing/bridges/objects) and place/orient it using OSM tags
4. Create "borders" using splines filled with object clusters, e.g. water hazard edges, course O.B. areas

### TODO

- ~~Trees from LiDAR~~ done
- ~~Water bodies & streams with procedurally-generated natural features~~ done
- ~~Assign tree types~~ done — a `natural=wood` polygon tagged `leaf_type=needleleaved` (or `broadleaved`) hints untyped trees inside it; per-theme species buckets in `course_output/tree_themes.json` map those to the theme's actual pine/deciduous asset ids (v2019) or `--tree-type-asset-path` (v2021+)
- ~~Calibrate tree size~~ done — placed scale = LiDAR-detected canopy height ÷ the prefab's measured native height (`asset_catalog.json` `native_height_m`, captured with `util/extract_tree_dimensions.py`); falls back to a course-relative remap for un-measured prefabs / non-rustic themes
- Assign vegetation using customized OSM ways
- Assign heavy rough to clear built-in natural objects and create OB
