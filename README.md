# vbn8.github.io

Local image comparison app backed by `bridge.py` and the `BWSI-CubeSat` comparison pipeline.


## Run locally

1. Create and activate the virtual environment:
    - **Windows**: `python -m venv venv && venv\Scripts\activate`
    - **macOS/Linux**: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip -r requirements.txt`
3. Start the app: `python bridge.py`
4. Open `http://localhost:8000/index`

`index.html` should be loaded through `bridge.py`, not opened directly from disk, because the page posts uploaded files to the backend route at `/compare`.