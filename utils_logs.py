# utils_logs.py
import io
import contextlib

class LogCapture:
    def __init__(self):
        self.buffer = io.StringIO()

    def __enter__(self):
        self._stdout = contextlib.redirect_stdout(self.buffer)
        self._stderr = contextlib.redirect_stderr(self.buffer)
        self._stdout.__enter__()
        self._stderr.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stderr.__exit__(exc_type, exc, tb)
        self._stdout.__exit__(exc_type, exc, tb)

    def text(self):
        return self.buffer.getvalue()
