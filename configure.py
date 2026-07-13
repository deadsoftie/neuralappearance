# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# Ensure setuptools is installed
if importlib.util.find_spec('setuptools') is None:
    print('\nInstall setuptools and restart')
    print('------------------------------')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'setuptools'])
    subprocess.call([sys.executable, *sys.argv], shell=True)
    exit(0)

# Ensure commentjson is installed
try:
    import commentjson
except ImportError:
    print('\nInstall commentjson and restart')
    print('------------------------------')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'commentjson'])
    subprocess.call([sys.executable, *sys.argv], shell=True)
    exit(0)

PROJECT_DIR = Path(__file__).resolve().parent

# Add falcor2 tools external path
sys.path.append(str(PROJECT_DIR / 'external/falcor2/tools'))
INTERACTIVE = True


def get_os():
    """
    Return the OS name (windows, linux, macos).
    """
    platform = sys.platform
    if platform == 'win32':
        return 'windows'
    elif platform == 'linux' or platform == 'linux2':
        return 'linux'
    elif platform == 'darwin':
        return 'macos'
    else:
        raise NameError(f'Unsupported OS: {sys.platform}')


def get_platform():
    """
    Return the platform name (x86_64, aarch64).
    """
    machine = platform.machine()
    if machine == 'x86_64' or machine == 'AMD64':
        return 'x86_64'
    elif machine == 'aarch64' or machine == 'arm64':
        return 'aarch64'
    else:
        raise NameError(f'Unsupported platform: {machine}')


def run_command(
    command: str | list[str],
    shell: bool = True,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
):
    """
    Helper to run a command, stream output to console and still allow it to
    be returned
    """
    if isinstance(command, str):
        command = [command]
    if get_os() == 'windows':
        command[0] = command[0].replace('/', '\\')
    print(' '.join(command))
    sys.stdout.flush()
    if shell:
        command = ' '.join(command)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
        shell=shell,
        env=env,
        cwd=cwd,
    )
    assert process.stdout is not None
    out = ''
    while True:
        nextline = process.stdout.readline()
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
        out += nextline
    process.communicate()
    return process.returncode, out


def git(args: str, cwd: Path | None = None):
    if cwd is None:
        cwd = PROJECT_DIR
    command = 'git ' + args
    rc, _ = run_command(command, shell=True, cwd=cwd)
    return rc


def is_system_python_environment():
    if hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        return False
    if os.environ.get('VIRTUAL_ENV'):
        return False
    if os.environ.get('CONDA_DEFAULT_ENV') and os.environ.get('CONDA_DEFAULT_ENV') != 'base':
        return False
    return True


def get_python_env():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(PROJECT_DIR)
    return env


def setup_submodules():
    """Sync and update all submodules recursively."""
    rc = git('submodule sync --recursive')
    if rc != 0:
        print('Failed to sync submodules.')
        exit(1)
    rc = git('submodule update --recursive --init')
    if rc != 0:
        print('Failed to update submodules.')
        exit(1)


def setup_falcor_2(nobuild: bool = False, vs2022: bool = False):
    setup_submodules()

    print('\nInstall editable falcor2/slangpy packages + dependencies (without building)')
    print('------------------------------')
    env = os.environ.copy()
    env['NO_CMAKE_BUILD'] = '1'
    rc = subprocess.call(
        'pip install -r external/falcor2/requirements-dev.txt', shell=True, env=env
    )
    if rc != 0:
        print('Failed to install falcor2 dependencies.')
        exit(1)
    rc = subprocess.call(
        'pip install --editable external/falcor2/external/slangpy', shell=True, env=env
    )
    if rc != 0:
        print('Failed to install slangpy.')
        exit(1)
    rc = subprocess.call('pip install --editable external/falcor2', shell=True, env=env)
    if rc != 0:
        print('Failed to install falcor2.')
        exit(1)

    if vs2022:
        print('\nConfigure falcor2 for Visual Studio 2022')
        print('------------------------------')
        try:
            sys.path.insert(0, str(PROJECT_DIR / 'external/falcor2/tools'))
            import build

            build.configure('windows-vs2022')
        except Exception as e:
            print(f'Failed to configure for VS2022: {e}')
            exit(1)
    elif not nobuild:
        print('\nRun falcor2 build (includes SlangPy)')
        print('------------------------------')
        try:
            sys.path.insert(0, str(PROJECT_DIR / 'external/falcor2/tools'))
            import build

            preset = build.get_default_preset()
            build.configure(preset)
            build.build(preset)
        except Exception:
            print('Failed to build with Ninja (are you in an x64 Native Tools command prompt?).')
            exit(1)

    print('\nInstall neural material python requirements')
    print('------------------------------')
    rc = subprocess.call('pip install -r requirements.txt', shell=True)
    if rc != 0:
        print('Failed to install requirements.')
        exit(1)

    print('\nInstallation complete')
    print('------------------------------')
    print('Editable slangpy package installed in external/falcor2/external/slangpy')
    print('Editable falcor2 package installed in external/falcor2')
    if vs2022:
        sln_path = 'external/falcor2/build/windows-vs2022/falcor2.sln'
        print('\nNOTE: VS2022 solution generated. Open and build the solution at:')
        print(f'  {sln_path}')
    elif nobuild:
        print(
            '\nNOTE: Build was skipped (--nobuild). You will need to build falcor2 manually using cmake or vs2022'
        )


def setup_vscode_directory():
    """
    Ensure .vscode/settings.json and .vscode/launch.json exist and contain
    all defaults from .vscode-default, without overwriting user modifications.
    """
    parent_dir = PROJECT_DIR
    vscode_dir = parent_dir / '.vscode'
    vscode_default_dir = parent_dir / '.vscode-default'
    vscode_dir.mkdir(exist_ok=True)
    changed = False
    changed |= merge_vscode_settings(
        vscode_default_dir / 'settings.json',
        vscode_dir / 'settings.json',
    )
    changed |= merge_vscode_launch_configs(
        vscode_default_dir / 'launch.json',
        vscode_dir / 'launch.json',
    )
    if changed:
        print('VS Code workspace defaults updated.')


def write_text_if_valid(path: Path, raw_text: str):
    """
    Write text to a file only if it can be parsed as JSON (ignoring comments).
    Raises ValueError if the text is not valid JSON.
    """
    try:
        commentjson.loads(raw_text)
        path.write_text(raw_text, encoding='utf-8')
    except Exception as e:
        print(f'Failed to merge {path}: {e}')


def merge_vscode_settings(default_path: Path, user_path: Path) -> bool:
    """
    Merge default settings into user settings.json, adding missing top-level
    keys without overwriting existing ones. Preserves user's existing file
    content (including comments). Returns True if the file was modified.
    """
    if not default_path.exists():
        return False
    defaults = commentjson.loads(default_path.read_text(encoding='utf-8'))
    if user_path.exists():
        raw_text = user_path.read_text(encoding='utf-8')
        user = commentjson.loads(raw_text)
    else:
        raw_text = '{}\n'
        user = {}
    missing = {k: v for k, v in defaults.items() if k not in user}
    if not missing:
        return False
    lines = []
    for key, value in missing.items():
        lines.append(f'    {json.dumps(key)}: {json.dumps(value)}')
    snippet = '\n' + ',\n'.join(lines) + ','
    pos = raw_text.index('{')
    raw_text = raw_text[: pos + 1] + snippet + raw_text[pos + 1 :]
    write_text_if_valid(user_path, raw_text)
    return True


def merge_vscode_launch_configs(default_path: Path, user_path: Path) -> bool:
    """
    Merge default launch configurations into user launch.json, adding missing
    configurations (matched by name) without overwriting existing ones.
    Preserves user's existing file content (including comments).
    Returns True if the file was modified.
    """
    if not default_path.exists():
        return False
    defaults = commentjson.loads(default_path.read_text(encoding='utf-8'))
    if user_path.exists():
        raw_text = user_path.read_text(encoding='utf-8')
        user = commentjson.loads(raw_text)
    else:
        raw_text = json.dumps({'version': '0.2.0', 'configurations': []}, indent=4) + '\n'
        user = {'version': '0.2.0', 'configurations': []}
    existing_names = {c['name'] for c in user.get('configurations', [])}
    to_add = [c for c in defaults.get('configurations', []) if c['name'] not in existing_names]
    if not to_add:
        return False
    fragments = []
    for config in to_add:
        fragments.append('        ' + json.dumps(config, indent=4).replace('\n', '\n        '))
    snippet = '\n' + ',\n'.join(fragments) + ','
    match = re.search(r'"configurations"\s*:\s*\[', raw_text)
    if not match:
        raise ValueError("Could not find 'configurations' array in launch.json")
    pos = match.end()
    raw_text = raw_text[:pos] + snippet + raw_text[pos:]
    write_text_if_valid(user_path, raw_text)
    return True


def install(nobuild: bool = False, vs2022: bool = False):
    setup_vscode_directory()

    # Check for system python environment
    if INTERACTIVE:
        if is_system_python_environment():
            print(
                """WARNING: It looks like you're running from a system python environment. falcor2 and SlangPy both install editable
packages that are likely to function much better within a contained virtual environment, created with venv or conda.
Are you sure you want to continue in the system python environment? (y/n): """
            )
            if input().lower() not in ['y', 'yes']:
                print('Aborting setup. Please run the script again within a virtual environment.')
                sys.exit(0)

    # Check we can find git
    try:
        subprocess.check_output('git --version', shell=True)
    except subprocess.CalledProcessError:
        print('Cannot get git version! Please ensure git is on the PATH.')
        sys.exit(1)

    # Public SlangPy pulls a data submodule with Git LFS-managed assets.
    try:
        subprocess.check_output('git lfs version', shell=True)
    except subprocess.CalledProcessError:
        print("Cannot get git-lfs version! Please install Git LFS and run 'git lfs install'.")
        sys.exit(1)

    setup_falcor_2(nobuild=nobuild, vs2022=vs2022)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Setup the NeuralAppearance repository.')
    parser.add_argument(
        '--silent',
        '-s',
        action='store_true',
        help='Run the setup without any need for interactive input',
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    install_parser = subparsers.add_parser('install', help='Setup the repository')
    install_parser.add_argument(
        '--nobuild', action='store_true', help='Skip the falcor2/SlangPy CMake build step'
    )
    install_parser.add_argument(
        '--vs2022',
        action='store_true',
        help='Generate a Visual Studio 2022 solution instead of building with Ninja (Windows only)',
    )

    args = vars(parser.parse_args())
    INTERACTIVE = not args['silent']

    if args['command'] == 'install':
        if args['vs2022'] and get_os() != 'windows':
            print('Error: --vs2022 is only supported on Windows.')
            sys.exit(1)
        if args['vs2022'] and args['nobuild']:
            print('Error: --vs2022 and --nobuild are mutually exclusive.')
            sys.exit(1)
        install(nobuild=args['nobuild'], vs2022=args['vs2022'])
