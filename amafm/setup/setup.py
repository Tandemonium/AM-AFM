import os
from typing import Optional


def setup_venv(cuda: str='cu118', venv_name: str='.venv', venv_dir: Optional[str]=None, 
               python_path: Optional[str]=''):
    """
        Parameters:
            cuda: Cuda version for toch installation (e.g. 'cu118' or 'cpu')
            venv_name: Name of the virtual environment.
            venv_dir: Directory where to place the virtual environment (without trailing '/').
                Defaults to working directory.
            python_path: Path to directory of Python version (without trailing '/').
                Defaults to default Python version set in PATH.
    """

    if venv_dir is None:
        venv_dir = os.getcwd()
    if os.name == 'nt':
        os.system(rf'setup\setup_venv_windows.sh {cuda} "{venv_dir}" "{venv_name}" {python_path}')
    elif os.name == 'posix':
        os.system(f'sh setup/setup_venv_posix.sh {cuda} "{venv_dir}" "{venv_name}" {python_path}')
