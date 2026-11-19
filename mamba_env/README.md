# Environment Setup

To load the environment, follow these steps:

### Load the Mamba module

```bash
ml mamba
```

### Create the environment

Create an environment named `cse494` using the provided `environment.yml` file:

```bash
mamba env create -n cse494 --file /path/to/environment.yml
```

**Note:** Installing the packages may take a while.

### Activate the environment

```bash
source activate cse494
```

### Run Python commands

You can now run your Python scripts and commands using:

```bash
python your_script.py
```