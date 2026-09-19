import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from eeg_decode.config import settings

if __name__ == "__main__":
    uvicorn.run("eeg_decode.app:app", host=settings.host, port=settings.port, reload=False)
