# vbn8.github.io

Local image comparison app backed by `bridge.py` and the `BWSI-CubeSat` comparison pipeline.


## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Start the app: `.venv/Scripts/python bridge.py`
3. Open `http://localhost:8000/index`

## On the cloud

Visit `xvb88.pythonanywhere.com/index`

`index.html` should be loaded through `bridge.py`, not opened directly from disk, because the page posts uploaded files to the backend route at `/compare`.