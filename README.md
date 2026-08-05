This is a PGA2k utility to generate .course files from USGS LiDAR and OpenStreetView OSM, based on Chad Rockey's excellent [TCG Designer Tools](https://github.com/chadrockey/TGC-Designer-Tools).

#How-To

This Python app needs the following libs:

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
2. Draw your course for OSM using [OpenStreetMap](https://openstreetmap.org/). Tag everything for golf - fairways, holes, etc.
3. Open the app & select your working folder.
4. Click "Init" to initialize the working folder. This adds some subfolders and a project file.
5. Put your LAZ files into "laz".
6. Click "Ingest LAZ" to load the LiDAR point cloud.
7. Console will output the extent coordinates. Use them to download OSM from OpenStreetMap. Move the OSM file to the working folder.
8. Click "Ingest OSM" to load the course map.
9. Click "Ingest Course" to unpack a .course file. Start from a blank that you created in TCG 2019. (TODO: add template to source)
10. Click "Generate Terrain" to do a coarse (~500 stamp) hex grid to roughly match LAZ.
11. Click "Refine Terrain" to create smaller stamps to add detail. Play with the settings for best results; rollover for details. Use mask to refine selected areas, and adjust buffer to include or exclude areas. Select mask objects under "Splines" tab.
12. Shortcuts: mouse wheel toggles preview versions; shift-wheel changes preview; ctrl-wheel zooms; wheel click moves map.
13. Click "Write Terrain" to create the heightmap.
14. Click "Write Splines" to generate course objects.
15. Under "Splines", filter by "hole" to select up to 18 holes for your course (apply Mask to exclude holes). Click "Write Holes" to add holes.
16. Click "Repack" to output the .course file to the working directory (give it a name first; ".course" suffix is not needed).
17. Click "Copy to Game Folder" to move the .course file to the appropriate PGA2K directory. Only v2019 is supported but .course can be opened in v2021. (TODO: compatibility)

#TODO

- Trees from LiDAR
- Water bodies & streams with procedural-generated nature
- Assign tree types & vegetation using customized OSM ways
- Assign heavy rough to clear built-in natural objects and create OB
