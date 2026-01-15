
# HotelQuEST: Balancing Quality and Efficiency in Agentic Search

HotelQuEST is a benchmark for evaluating agentic search systems, focusing on the trade-off between **answer quality** and **computational efficiency**.

---

## 1. Environment Setup

### 1.1 Install Miniconda (if you don’t have Conda)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```


### 1.2 Create the Conda Environment
Make sure you are in the project root directory (where environment.yml is located):
```bash
conda env create -f environment.yml
```


### 1.3 GPU Drivers (Ubuntu) 
If you are on an Ubuntu machine with an NVIDIA GPU, install the recommended drivers (if needed):
```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```


After the machine reboots, verify that the drivers are correctly installed:
```bash
nvidia-smi
```

## 2. Prepare the Description Index
This script builds the description index used by the agentic search pipeline.
```bash
python prepare_description_index.py
```

## 3. Prepare the Reviews Index
HotelQuEST uses a Milvus server as the vector database for storing review embeddings due to their large scale. To run Milvus using Docker:

```bash
sudo apt install docker.io -y
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

This will start a standalone Milvus instance with embedding support.

```bash
python prepare_reviews_index.py
```


## 4. Run the agent!


```bash
python run_agent.py --input final_benchmark.csv --output answers.csv
```




## 5. Additional Notes

The **`notebooks/`** directory contains all notebooks used for experiments, analysis, and the full evaluation pipeline.

### Running the Evaluation
To evaluate the agent responses, you must use **Arize Phoenix**:

1. Create a new Phoenix experiment.
2. Copy the experiment URI.
3. Paste it into the dataset configuration line in the evaluation notebook or script.

### Index Preparation
In the index preparation scripts, you can select:

- The **embedding model** you want to use.
- The **index type** (e.g., HNSW, IVF, etc.)

Choose these according to your hardware capacity.  
**Note:** The raw hotel reviews dataset exceeds **20GB**, so ensure you have enough memory and disk space.

### LLM Configuration
You can configure which LLM the agent uses by editing **`llm.py`**.

Different LLMs — even when accessed through AWS **Bedrock** — may produce slightly different output formats.  
Make sure to adjust any parsing logic accordingly.

