# PCLA Evaluation and Analysis

This repository contains the setup for benchmarking the [PCLA](https://github.com/MasoudJTehrani/PCLA) agents on [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive), the evaluation results, and the materials from our analysis of agent features and driving performance. These include the feature catalogue, statistical analyses, tables and figures.

You can run the 40 PCLA agents used in our study on the original Bench2Drive 220-route suite, using [CARLA](https://github.com/carla-simulator/carla) 0.9.15.

## 1. Install PCLA

Use Linux with an NVIDIA GPU, working graphics/Vulkan drivers, Git and Conda. Clone this repository and enter its folder:

```bash
git clone https://github.com/LeonardoPrioriello/PCLA_Eval.git
cd PCLA_Eval
```

Use this repository's folder as the project root, with PCLA, Bench2Drive and CARLA installed inside it:

```text
PCLA_Eval/
├── run.py
├── bridge.py
├── prepare.py
├── patches/
├── PCLA/
├── Bench2Drive/
└── CARLA/
```

Starting from the project root, follow PCLA's [setup instructions](https://github.com/MasoudJTehrani/PCLA#1-setup) and [weight download instructions](https://github.com/MasoudJTehrani/PCLA#2-pre-trained-weights), including its Conda environment, assets, torch-scatter, CUDA Toolkit and the additional steps for the LMDrive agent. This creates the `PCLA/` folder inside the project. Download the weights for the agents you intend to run.

For CARLA, follow step 3 below to install **0.9.15** and its maps. Step 4 also sets the Python client to **0.9.15**, replacing the version installed by PCLA's environment.

If LMDrive's LAVIS installation fails because `virtualenv` requires a newer `typing-extensions` than the one installed by PCLA, rerun that step with `python setup.py develop --no-deps` in the **LAVIS** directory, then complete the remaining PCLA instructions.

## 2. Download Bench2Drive

Return to the project root (the folder containing `run.py`) and download the Bench2Drive release that uses CARLA 0.9.15:

```bash
git clone --branch 0.0.3 https://github.com/Thinklab-SJTU/Bench2Drive.git
```

## 3. Install CARLA and its maps

From the same directory, install CARLA **0.9.15** and its additional maps:

```bash
mkdir CARLA
cd CARLA
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xf CARLA_0.9.15.tar.gz
wget -P Import https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
bash ImportAssets.sh
cd ..
```

## 4. Apply the compatibility fixes

From the project root, match the Python client to CARLA 0.9.15 and apply the fixes for MindDrive agent weight loading and Bench2Drive's Python 3.10 route parser:

```bash
conda activate PCLA
python -m pip install carla==0.9.15
python prepare.py
```

## 5. Test one route

List the available agents:

```bash
python run.py --list-agents
```

In the commands below, replace `<agent>` with a name from this list, without the angle brackets. To test one route, run:

```bash
python run.py <agent> --route 1711 --output results/<agent>-test
```

The launcher starts CARLA. For example, to test TransFuser V6 RegNet:

```bash
python run.py tfv6_regnet --route 1711 --output results/tfv6_regnet-test
```

## 6. Run all 220 routes

```bash
python run.py <agent>
```

If you interrupt the benchmark or execution stops unexpectedly, resume by repeating the same command for that agent:

```bash
python run.py <agent>
```

The launcher keeps the saved results and continues from the first route without a recorded result. Keep the same `--route` and `--output` options if you specified them in the original command. Use a new `--output` directory, instead, for a separate run.

By default, results are saved in `results/<agent>/results.json` and execution messages in `results/<agent>/evaluator.log`. If you specify `--output`, both files are saved in that directory.

CARLA may sometimes crash, after many routes executed sequentially, or during some routes in particular for some agents; if this happens with a segmentation fault, the launcher automatically restarts CARLA and retries that route, allowing three attempts in total per route. After three such crashes without a result, it records a manual skip and continues to the next route.