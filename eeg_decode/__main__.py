from eeg_decode.app import app
from eeg_decode.config import settings

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("eeg_decode.app:app", host=settings.host, port=settings.port, reload=False)
