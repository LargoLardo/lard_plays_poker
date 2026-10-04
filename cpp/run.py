"""Build/run the native trainer or checks on Windows, macOS, and Linux."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

CPP = Path(__file__).resolve().parent


def build(test=False, *, arena=False):
    name = 'arena' if arena else 'test' if test else 'train'
    directory = CPP / 'build'
    directory.mkdir(exist_ok=True)
    suffix = '.exe' if os.name == 'nt' else ''
    binary = directory / (name + suffix)
    sources = [CPP / 'poker.hpp', CPP / 'trainer.cpp', Path(__file__)]
    if test or arena:
        sources.append(CPP / (name + '.cpp'))
    if binary.exists() and binary.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in sources):
        return binary
    choices = ('cl', 'clang-cl', 'g++', 'clang++') if os.name == 'nt' else ('c++', 'g++', 'clang++')
    configured = os.environ.get('CXX')
    compiler = [part.strip('"') for part in shlex.split(configured, posix=os.name != 'nt')] if configured else [next((p for p in choices if shutil.which(p)), '')]
    if not compiler or not compiler[0]:
        raise RuntimeError('Install a C++17 compiler. On Windows, use a Visual Studio Developer Command Prompt, or set CXX to g++/clang++.')
    temporary = directory / (name + '.tmp' + suffix)
    source = CPP / (name + '.cpp' if test or arena else 'trainer.cpp')
    if Path(compiler[0]).stem.lower() in ('cl', 'clang-cl'):
        flags = ['/nologo', '/std:c++17', '/O2', '/EHsc', '/W4', str(source),
                 '/Fe:' + str(temporary), '/Fo:' + str(directory / (name + '.obj'))]
        if not test:
            flags.append('/DNDEBUG')
    else:
        flags = ['-std=c++17', '-O3', '-pthread', '-Wall', '-Wextra', '-Wpedantic', str(source), '-o', str(temporary)]
        if not test:
            flags.append('-DNDEBUG')
    subprocess.run(compiler + flags, check=True, stdout=sys.stderr)
    os.replace(temporary, binary)
    return binary


if __name__ == '__main__':
    arguments = sys.argv[1:]
    test = bool(arguments and arguments[0] == '--test')
    if test:
        arguments.pop(0)
    try:
        command = [str(build(test)), *arguments]
        if os.name != 'nt':
            os.execv(command[0], command)
        # Windows console events also reach the child; let it finish/save its batch.
        process = subprocess.Popen(command)
        while True:
            try:
                sys.exit(process.wait())
            except KeyboardInterrupt:
                continue
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
