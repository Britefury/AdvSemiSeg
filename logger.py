import sys

class LogStream (object):
    def __init__(self, path, stream):
        self.path = path
        self.stream = stream

    def write(self, x):
        with open(self.path, 'a+') as f_out:
            f_out.write(x)
        self.stream.write(x)

    def flush(self):
        self.stream.flush()


class LogFile (object):
    def __init__(self, path):
        self.log_path = path
        if self.log_path is not None:
            self.__stdout = LogStream(self.log_path, sys.stdout)
            self.__stderr = LogStream(self.log_path, sys.stderr)


    def connect_streams(self):
        if self.log_path is not None:
            sys.stdout = self.__stdout
            sys.stderr = self.__stderr

    def disconnect_streams(self):
        if self.log_path is not None:
            sys.stdout = self.__stdout.stream
            sys.stderr = self.__stderr.stream


    def __enter__(self):
        self.connect_streams()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect_streams()
        return self



