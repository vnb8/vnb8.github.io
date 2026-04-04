# vbn8.github.io

Local image comparison app backed by `bridge.py` and the `BWSI-CubeSat` comparison pipeline.

## GitHub Pages

GitHub Pages can host `index.html`, but it cannot run `bridge.py`. For the published page to work, deploy `bridge.py` somewhere that can run Python and set `window.IMAGE_COMPARE_API_BASE` in `index.html` to that backend origin, for example `https://your-backend.example.com`.

The page will then call `POST /compare` on that backend instead of assuming the API is on the same origin.

## Run locally

1. Create and activate the virtual environment.
2. Install dependencies: `pip install flask numpy opencv-python shapely`
3. Start the app: `python bridge.py`
4. Open `http://127.0.0.1:8000`

`index.html` should be loaded through `bridge.py`, not opened directly from disk, because the page posts uploaded files to the backend route at `/compare`.