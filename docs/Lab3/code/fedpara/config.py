MMAP_PATH: str = '/tmp'
SERVER_PARAMS_FILEDESC: str = 'fedavg_server_params.mmap'
CLIENT_PARAMS_FILEDESC: str = 'fedavg_client_{}_params.mmap'
CLIENT_SIGNAL_FILEDESC: str = 'fedavg_client_{}_signal.mmap'
CLIENT_INFO_FILEDESC: str = 'fedavg_client_{}_info.mmap'

SIG_INIT: int = 0x00
SIG_S_READY: int = 0x01
SIG_S_BUSY: int = 0x02
SIG_S_ERROR: int = 0x04
# SIG_S_QUERY: int = 0x03
SIG_S_CLOSE: int = 0x0f
SIG_C_READY: int = 0x10
SIG_C_BUSY: int = 0x20
SIG_C_ERROR: int = 0x40
SIG_C_CLOSE: int = 0xf0
