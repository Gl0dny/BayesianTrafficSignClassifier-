import os
import subprocess
import sys
import venv

def create_virtualenv(env_name):
    venv.create(env_name, with_pip=True)
    print(f"Virtual environment '{env_name}' created.")

def get_installed_packages(env_name):
    if os.name == 'nt':
        pip_executable = os.path.join(env_name, 'Scripts', 'pip')
    else:
        pip_executable = os.path.join(env_name, 'bin', 'pip')
    result = subprocess.run([pip_executable, 'list'], capture_output=True, text=True)
    packages = result.stdout.splitlines()[2:]  # Skip the first two lines (header)
    return packages

def install_requirements(requirements_file, env_name):
    if os.name == 'nt':
        pip_executable = os.path.join(env_name, 'Scripts', 'pip')
    else:
        pip_executable = os.path.join(env_name, 'bin', 'pip')
    subprocess.check_call([pip_executable, 'install', '-r', requirements_file])
    print("Requirements installed.")

def run_script(script_name, env_name):
    if os.name == 'nt':
        python_executable = os.path.join(env_name, 'Scripts', 'python')
    else:
        python_executable = os.path.join(env_name, 'bin', 'python')
    subprocess.check_call([python_executable, script_name])
    print("Script executed.")

def main():
    env_name = "venv"
    requirements_file = "setup/requirements.txt"
    script_name = "control/main.py"

    create_virtualenv(env_name)
    print(f"Running inside virtual environment '{env_name}'.")

    # Check installed packages before installing new ones
    print("Installed packages before installing new ones:")
    packages_before = get_installed_packages(env_name)
    for package in packages_before:
        print(package)

    install_requirements(requirements_file, env_name)

    # Check installed packages after installing new ones
    print("\nInstalled packages after installing new ones:")
    packages_after = get_installed_packages(env_name)
    for package in packages_after:
        print(package)

    run_script(script_name, env_name)

if __name__ == "__main__":
    main()
