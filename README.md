# vbn8.github.io

Local image comparison app backed by `bridge.py` and the `BWSI-CubeSat` comparison pipeline.


## Run locally

1. Create and activate the virtual environment.
2. Install dependencies: `pip install flask numpy opencv-python shapely`
3. Start the app: `python bridge.py`
4. Open `http://localhost:8000/index`

`index.html` should be loaded through `bridge.py`, not opened directly from disk, because the page posts uploaded files to the backend route at `/compare`.