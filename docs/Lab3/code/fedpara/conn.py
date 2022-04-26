import mmap
import pickle
import itertools
from typing import Any

class ConnABC(object):
    def __init__(self, path: str, size: int=0, mult:int=2) -> None:
        super().__init__()
        self.mult: int = mult # Multiplier of size. RealSize = Size * Multiplier
        self.path: str = path # Mapped path of shared memory
        self.size: int = size # Size of memory. When size==0, the connection will be set upon an existing file
        self.closed: bool = True
        self.open()

    def open(self):
        """Start connection
        """
        if self.closed:
            self._create_mmap()

    def _create_mmap(self) -> None:
        """Create mmap
        The server is responsible of creating mmap files. It must decide the size of share memory

        The client, on the other hand, open a mmap file directly. So size==0 on the client size, and the client should not create new file on disk
        """

        # Creating an empty file on disk
        if self.size > 0:
            with open(self.path, 'wb') as f:
                f.write(bytearray(itertools.repeat(0, int(self.size * self.mult))))
        
        # Open the file and mmap
        self.fd = open(self.path, 'r+b')
        self.mmap = mmap.mmap(self.fd.fileno(), 0, mmap.MAP_SHARED)
        self.size = self.mmap.size()
        self.closed = False
    
    
    def set(self, obj: Any, encode:bool=True) -> bool:
        """Set the content of shared memory to an object

        Args:
            obj (Any): bytes array or other types of object
            encode (bool, optional): Encode the object or not. Defaults to True.

        Raises:
            BufferError: The object exceeds size limit

        Returns:
            bool: Status
        """

        # If encode is True, encode the object with pickle
        if encode:
            obj_ser = pickle.dumps(obj)
        else:
            obj_ser = obj
        if len(obj_ser) > self.size:
            raise BufferError(f'Oversized object {len(obj_ser)} exceed limit of {self.size}')
        
        self.mmap.seek(0) # Remember to seek(0)
        self.mmap.write(obj_ser)
        return True

    def get(self, decode: bool=True) -> Any:
        """Get object from shared memory

        Args:
            decode (bool, optional): Decode the object or not. Defaults to True.

        Returns:
            Any: Result
        """
        self.mmap.seek(0) # Remember to seek(0)

        # # If decode is True, decode the object with pickle
        if decode:
            return pickle.loads(self.mmap.read())
        else:
            return self.mmap.read()

    
    def close(self):
        """Shut the connection down gracefully
        """
        if not self.mmap.closed:
            self.mmap.close()
        
        if not self.fd.closed:
            self.fd.close()
        
        self.closed = True