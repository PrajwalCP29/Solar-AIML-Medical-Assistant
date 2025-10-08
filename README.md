# Solar-AIML-Medical-Assistant

# AlpaCare Medical Instruction Assistant

**Submission for the AIML Assessment at Solar Industries India Ltd.**

---

<p align="center">
  <img src="https://placehold.co/600x300/e2e8f0/2d3748?text=Medical+AI+Assistant&font=inter" alt="Project Banner Image">
</p>


## Table of Contents
- [Project Overview](#project-overview)
- [Architecture and Approach](#architecture-and-approach)
- [Dataset](#dataset)
- [Instructions to Run](#instructions-to-run)
- [Project Deliverables](#project-deliverables)
- [Safety and Mitigation](#safety-and-mitigation)
- [Evaluation](#evaluation)
- [Limitations](#limitations)

## Project Overview
The goal of this project is to build a safe, non-diagnostic medical instruction assistant. This was achieved by fine-tuning a pre-trained Large Language Model (LLM) with less than 7 billion parameters on the `lavita/AlpaCare-MedInstruct-52k` dataset.

The project emphasizes safety, reproducibility, and efficient training by leveraging QLoRA (Quantized Low-Rank Adaptation) within a Google Colab environment. The final deliverable is not a hosted model but the trained adapter, supporting code, and a comprehensive report.

## Architecture and Approach
The technical approach was designed for efficiency and effectiveness, given the constraints of the project.

- **Base Model**: `mistralai/Mistral-7B-Instruct-v0.2`
  - **Justification**: This model was selected for its powerful instruction-following capabilities, a permissive Apache 2.0 license, and its proven performance-to-size ratio. It fits the `<7B parameter` constraint and is well-supported by the Hugging Face ecosystem.

- **Fine-Tuning Technique**: **QLoRA** (Quantized Low-Rank Adaptation)
  - The base model is loaded in 4-bit precision using the `bitsandbytes` library. This dramatically reduces the memory footprint, making it possible to train on a single T4 GPU available in Google Colab.
  - Small, trainable Low-Rank Adaptation (LoRA) adapters are then attached to the model's attention layers (`q_proj`, `v_proj`).
  - During training, only the adapter weights are updated, keeping the base model frozen. This is a highly parameter-efficient fine-tuning (PEFT) method.

- **Training Framework**: The `SFTTrainer` from the TRL (Transformer Reinforcement Learning) library was used to perform supervised fine-tuning. This trainer simplifies the process by handling data formatting, packing, and the training loop.

## Dataset
- **Primary Dataset**: `lavita/AlpaCare-MedInstruct-52k`, sourced directly from the Hugging Face Hub. This dataset contains 52,002 medical instruction-response pairs.

- **Preprocessing**:
  1. A unified prompt template (`### Instruction: ... ### Response: ...`) was applied to all samples to structure the data for the instruction-tuned model.
  2. Entries with a placeholder input (`<noinput>`) were handled to ensure a clean prompt structure.

- **Data Splitting**: The dataset was shuffled and split into three sets:
  - **Training Set**: 90% (46,801 samples)
  - **Validation Set**: 5% (2,601 samples)
  - **Test Set**: 5% (2,601 samples)

## Instructions to Run
This project can be fully reproduced using Google Colab.

### Prerequisites
- A Google Account.
- A Google Colab environment with a **T4 GPU** runtime (`Runtime > Change runtime type > T4 GPU`).

### Setup and Execution
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/](https://github.com/)[PrajwalCP29]/[Solar-AIML-Medical-Assistant].git
    cd [Solar-AIML-Medical-Assistant]
    ```

2.  **Upload Files to Colab**
    - Open a new Google Colab notebook.
    - On the left sidebar, click the "Files" icon.
    - Upload the `inference_demo.ipynb` notebook from the `notebooks/` directory.
    - Upload the entire `adapters/` directory, which contains the trained `alpacare-medical-adapter`.

3.  **Run the Demo Notebook**
    - Open `inference_demo.ipynb` in Colab.
    - The first cell will install all required dependencies from `requirements.txt`.
    - Subsequent cells will:
      - Load the base `mistralai/Mistral-7B-Instruct-v0.2` model in 4-bit precision.
      - Load the fine-tuned LoRA adapter from the uploaded `adapters/` directory.
      - Merge the adapter into the base model.
      - Provide example prompts to demonstrate the model's instruction-following ability and the mandatory safety disclaimer.

## Project Deliverables
The repository is structured with the following deliverables:

| File / Directory                     | Description                                                                 |
| ------------------------------------ | --------------------------------------------------------------------------- |
| `data_loader.py`                     | Python script for loading, preprocessing, and splitting the dataset.        |
| `notebooks/colab-finetune.ipynb`     | The complete, runnable notebook for the fine-tuning process.                |
| `notebooks/inference_demo.ipynb`     | A clean notebook for demonstrating inference with the trained adapter.      |
| `adapters/alpacare-medical-adapter/` | The saved QLoRA adapter files—the core training artifact.                   |
| `evaluation/human_evaluation.csv`    | The spreadsheet containing results from the human evaluation.               |
| `requirements.txt`                   | A list of all Python dependencies required to run the project.              |
| `REPORT.pdf`                         | The final, detailed project report (max 6 pages).                           |
| `README.md`                          | This file.                                                                  |

## Safety and Mitigation
Safety was a top priority and was addressed through several mechanisms:

1.  **Hard-coded Disclaimer**: The inference pipeline programmatically appends a disclaimer to **every** generated output. This is a non-negotiable step that ensures the user is always warned about the educational nature of the tool.
2.  **Forbidden Content**: The model was explicitly designed to avoid providing diagnoses, prescriptions, or clinical decision support. The training data focuses on general medical knowledge and instructions.
3.  **Human-in-the-Loop**: The model is intended as an assistant, not an autonomous agent. The final report and this README strongly recommend that a human clinician always be involved in any real-world application.

## Evaluation
The model was evaluated primarily through **human review**.

- **Methodology**: 30 diverse prompts from the held-out test set were used to generate responses.
- **Reviewers**: These responses were reviewed by medically-literate individuals (medical and biology students) with their qualifications stated in the evaluation sheet.
- **Rubric**: Responses were scored on a 1-5 scale for **Helpfulness**, **Safety**, and **Clarity**.
- **Results**: A summary of the findings is available in `REPORT.pdf`, and the raw data is in `evaluation/human_evaluation.csv`.

## Limitations
- **Knowledge Cutoff**: The model's knowledge is static and limited to its training data. It is not aware of medical research or developments published after its training date.
- **Potential for Hallucination**: As with all LLMs, there is a risk of generating factually incorrect or nonsensical information. The disclaimer is the primary mitigation for this risk.
- **Not a Medical Device**: This project is a proof-of-concept and is not a certified medical device. It should not be used for any clinical or diagnostic purposes.
```eof
