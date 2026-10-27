# Translating Radiology Findings into Plain Language: Evaluating FLAN-T5 and Custom PyTorch LoRA on LaymanRRG – Aidan Fassy – 44368133

## Executive Summary

Within healthcare and medicine, radiological reports have the reputation of being extremely technical, filled with extensive medical jargon, and very unfriendly for an unskilled person to read. As a result, patients who desire transparency when it comes to their healthcare are often left anxious, confused, and prone to misinterpretations. It stands that there is a need for a layperson to understand and interpret their report correctly.

It is for this reason that the objective of this project is to build a sequence-to-sequence NLP prototype that translates jargon-heavy, complex diagnostic findings into plain, accurate, and simple layperson-accessible language – ideally at the 6th-grade level. This project uses the ACL 2025 BioLaySumm (LaymanRRG) open-source benchmark dataset.

The `google/flan-t5-base` model (~248M parameters) is used with a custom PyTorch Low-Rank Adaptation (LoRA) layer, which is injected into the attention query ($W_q$) and value ($W_v$) projections. A goal of this investigation was also to avoid heavy parameter fine-tuning by freezing $99.64\%$ of the base weights and only updating a lightweight rank-8 bottleneck (~$884\text{K}$ parameters, $0.36\%$ of total).

However, a problem still remains: standard metrics like ROUGE-1, ROUGE-2, and ROUGE-L measure surface n-gram overlap, not the underlying clinical truth. As a result of this, a model can subvert the testing objectives by memorising normal radiological findings (e.g., "lungs are clear, no acute abnormality") while omitting crucial clinical details, some of which can be as severe as hallucinating non-existent tumors.

So, the current investigation is to evaluate whether LoRA fine-tuning is able to genuinely improve factual clinical fidelity over an untuned zero-shot baseline, or if it simply increases superficial linguistic fluency on memorised screening templates across both full and deduplicated splits.

## Architecture & Working Principles

### FLAN-T5: Encoder-Decoder Architecture

A FLAN-T5 model is fundamentally an *Encoder-Decoder* architecture, which provides a distinct architectural advantage over *decoder-only* causal models (such as GPT-2). Radiological narratives are scattered, non-linear, and conditional; a radiologist might note an acute abnormality in the first sentence, describe unremarkable anatomy for several lines, and then introduce critical diagnostic context in the final impression. Unlike decoder-only models that process text strictly left-to-right, FLAN-T5's bidirectional encoder allows every token to attend to all other tokens simultaneously without a causal mask. This allows the model to build a complete, contextual representation of the entire clinical case before any summary text is produced.

The causal decoder then acts as a grounded information bottleneck. During generation, it uses *cross-attention* to constantly query the encoder's memory states ($H_{\text{enc}}$) at each step. This mechanism forces each generated layperson phrase to remain directly anchored to actual source evidence, significantly reducing the risk of unconstrained hallucinations.

### Custom PyTorch LoRA

#### LoRA Mathematical Breakdown & Implementation Details

In `modules.py`, instead of updating the full $768 \times 768$ linear weight matrices in FLAN-T5, the `LoRALinear` layer factorises the weight updates into two smaller matrices:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

Here, $W_0 \in \mathbb{R}^{768 \times 768}$ represents the original pre-trained weight matrix from `google/flan-t5-base`. We freeze this matrix by setting `requires_grad = False`, preventing PyTorch from computing gradients or allocating the heavy AdamW momentum and variance buffers across all 248M base parameters. 

The low-rank update is handled by the parallel adapter path. The down-projection matrix $A \in \mathbb{R}^{8 \times 768}$ compresses the incoming 768-dimensional token representation down into an 8-dimensional bottleneck, and is initialised using Gaussian noise ($\sigma = 0.02$) to break symmetry across the bottleneck channels. The up-projection matrix $B \in \mathbb{R}^{768 \times 8}$ then maps this compressed representation back up to the base 768-dimensional space. Crucially, $B$ is initialised entirely to zeros. Because $B = 0$ at step 0, the adapter contribution evaluates to zero ($B A x = 0$), ensuring that the forward pass initially reduces to $h = W_0 x$. This guarantees that training starts with the exact representations of the pre-trained FLAN-T5 model, rather than injecting random perturbations into the attention heads. Finally, the update is scaled by a constant factor $\frac{\alpha}{r}$ (here $\frac{32}{8} = 4.0$), which balances the adapter's update magnitude against the base weights and ensures that altering the rank $r$ does not require re-tuning the optimizer's base learning rate.

### Model Architecture 
![System Architecture](architecture.png)
Figure 1: Diagram of the FLAN-T5 Model Architecture

#### FLAN-T5 Backbone

The base model used is `google/flan-t5-base` (~248M parameters). The architecture consists of 12 encoder blocks and 12 decoder blocks with an internal model dimension of $d_{\text{model}} = 768$, 12 attention heads ($d_k = d_v = 64$), a feed-forward expansion dimension of $d_{\text{ff}} = 2048$, Pre-LN RMSNorm, and relative position biases. 

The bidirectional encoder has the function of capturing the complete report without any causal masking; this is critical because radiological reports are interdependent across disparate sections of text. The autoregressive decoder generates patient-facing words sequentially, continually referencing the encoder's memory states ($H_{\text{enc}}$) via the cross-attention mechanism. Cross-attention is crucial because it ensures that the plain-language output is grounded in actual medical observations.

![FLAN-T5 Sequence-to-Sequence Architecture](flan_t5_backbone.png)
Figure 2: Diagram of the T5 Backbone of the project

#### Custom LoRA Layer
Rather than modifying all of the attention projections, the custom `LoRALinear` layers are injected into the query ($W_q$) and value ($W_v$) projections across:
* **Encoder Self-Attention:** Steers attention across the scattered clinical findings to build a rich contextual representation.
* **Decoder Self-Attention:** Ensures that there is linguistic fluency, sentence structure, and a layperson tone across generated summary tokens.
* **Decoder Cross-Attention:** The core translational bridge. Allows the decoder to query important technical details in the encoder ($H_{\text{enc}}$) and extract the plain-language equivalents.

#### Parameter Counts & Optimiser Memory Analysis
**Base Model Parameters:** $247,577,856$ (frozen, `requires_grad = False`)  
**Trainable LoRA Parameters:** $884,736$ ($0.36\%$ of total)  
**Total Parameters:** $248,462,592$

Because $99.64\%$ of the weights are frozen, the AdamW optimizer only allocates gradient history, momentum ($m_t$), and second-moment variance ($v_t$) states for 884K parameters (~$7$ MB) instead of 248M parameters (~$2$ GB). This reduced peak training VRAM on an A100 GPU from $10.29$ GB during full fine-tuning down to $6.59$ GB with LoRA.

## Data Pipeline & Preprocessing
### Dataset
The `ACL 2025 BioLaySumm Subtask 2.1` dataset was used as a benchmark. This dataset consists of two distinct columns:
* `radiology_report`: This is the source column, which serves as the expert report, written by radiologists.
* `layman_report`: This is the target column. It is a plain-language summary, aimed at around the 6th-grade reading level.

The dataset has split sizes of $150,454$ for training, $10,000$ for validation, and $10,537$ for testing. Because the official shared-task test split is reserved for blind workshop evaluation, and in accordance with the course benchmarking guidelines, all quantitative and clinical safety evaluations are performed on the held-out validation split across both its full (`Val_Full`) and SHA-256 deduplicated (`Val_Unseen`) cohorts.

### Preprocessing & Tokenisation

In `dataset.py`, each input report is prepended with the task prompt `"Translate this radiology report into plain language for the patient: "` to activate FLAN-T5's instruction-following mode. Source narratives are capped at $512$ tokens, which covers over $95\%$ of full clinical reports without truncation, while target summaries are limited to $128$ tokens.

To prevent padding tokens from skewing training, `LaymanRRGDataSet.__getitem__` handles target padding explicitly. While T5 assigns `pad_token_id = 0`, PyTorch's `CrossEntropyLoss` expects `ignore_index = -100` to mask tokens out of the loss calculation:

```python
labels[labels == self._tokeniser.pad_token_id] = -100
```
### Leakage Prevention

A major issue with routine radiology data is that normal screening reports frequently repeat identical template phrasing (e.g., *"Lungs are clear. No acute cardiopulmonary abnormality."*). If these exact templates leak into both the training and validation sets, a model can easily game lexical metrics like ROUGE by simply memorising common phrases rather than learning how to actually translate clinical text.

To catch this, `load_and_split_dataset` normalises each report and computes its SHA-256 hash across the training split. The model is then evaluated across two validation sets: `Val_Full`, which serves as the official untouched validation split (10,000 cases) for standard competition benchmarking, and `Val_Unseen`, a filtered subset containing only cases whose normalised text never appeared anywhere in the training data, allowing us to directly test whether the model is genuinely generalising or just memorising templates.

## Midpoint Feasibility Review

### User Need, Scope & Acceptance Criteria
The intended users for this product are digital health engineering teams developing automated patient portal communication tools to translate dense, technical radiology reports into plain-language summaries at approximately a 6th-grade reading level.

The following acceptance criteria are established for this prototype:
* **Lexical Performance:** Achieve a gain of $\geq 10\%$ in ROUGE-1, ROUGE-2, and ROUGE-L over an untuned zero-shot baseline on the validation split.
* **VRAM Ceiling:** Keep peak GPU memory allocation strictly below $8.0$ GB during LoRA training (batch size $\geq 8$, sequence length 512) and below $4.0$ GB during evaluation on a single GPU node.
* **Clinical Safety Constraint:** Strictly zero hallucinations of acute findings and zero omissions of critical incidental findings (such as an apical pneumothorax or suspicious pulmonary nodule), audited across representative validation cases.
* **Throughput:** Achieve $\geq 30$ tokens/second during inference on Rangpur HPC GPU nodes.
* **Generalisation Check:** Keep the performance drop $\leq 15\%$ on an audited, deduplicated validation split (`Val_Unseen`) compared to the full split (`Val_Full`) to verify genuine generalisation over template memorisation.

### Model Choice & Course Concepts
This project uses the ACL 2025 BioLaySumm `LaymanRRG` open-source benchmark dataset. Inputs are prepended with the task conditioning prompt: `"Translate this radiology report into plain language for the patient: "`.

The selected architecture is `google/flan-t5-base` (~248M parameters), an encoder-decoder sequence-to-sequence transformer consisting of:
* **Encoder & Decoder Layers:** 12 encoder blocks and 12 decoder blocks
* **Model Dimensions:** $d_{\text{model}} = 768$, feed-forward expansion dimension $d_{\text{ff}} = 2048$
* **Attention Heads:** 12 heads with key/value dimension $d_k = d_v = 64$
* **Vocabulary Size:** 32,128 tokens
* **Normalisation:** Pre-LN RMSNorm with relative position biases

This project falls under the `[Hard Difficulty]` band. FLAN-T5's bidirectional encoder grounds plain-text translations in complex, non-linear findings, while the custom PyTorch Low-Rank Adaptation (LoRA) avoids the memory overhead of full fine-tuning. This setup directly investigates the core course dilemma: preventing models from earning deceptively high ROUGE scores by repeating boilerplate normal screening templates while failing to convey acute clinical findings.

### Preliminary Feasibility Evidence
Initial profiling of the `LaymanRRG` dataset confirmed that over $95\%$ of reports fit within the $512$-token limit without cutting off clinical findings. Exact string matching flagged repeated normal screening templates across splits, confirming the need for a deduplicated test set (`Val_Unseen`) to catch pure memorisation. A smoke test on an interactive Rangpur GPU node confirmed that `flan-t5-base` loads and runs autoregressive generation in pure PyTorch with zero issues. Finally, running a forward and backward pass with LoRA on an initial test batch peaked at $\sim 3.4$ GB VRAM on a single GPU. This left ample headroom below the $8.0$ GB ceiling, verifying that training would operate comfortably within our memory budget.

### Risks, Budget & Fallback
Key technical risks and mitigations include:
* **VRAM Out-of-Memory:** Full fine-tuning risks OOM errors due to AdamW optimizer states across 248M parameters on 512-token sequences. Custom LoRA is implemented as the primary strategy to restrict updates to $<0.4\%$ of parameters.
* **Boilerplate Inflation:** High ROUGE scores masking factual hallucinations or omissions of acute pathologies. Addressed through qualitative clinical safety autopsies.
* **Computing Budget:** $\sim 3$ GPU hours for data auditing and pipeline smoke tests; $\sim 8\text{--}10$ GPU hours total for training and benchmarking runs via Slurm on Rangpur.
* **Next Planned Experiment:** Launch a 1-epoch run of the custom PyTorch LoRA training script on Rangpur to verify monotonic loss decrease, tracking peak VRAM allocation and step latency.
* **Workable Fallback:** If memory allocation issues persist under full fine-tuning, reduce batch size to 4 with 4 gradient accumulation steps under FP16/bfloat16. If resource constraints remain intractable, fall back to the $\sim 60\text{M}$ parameter `flan-t5-small` model as the comparative baseline while retaining `flan-t5-base` with LoRA as the primary architecture.

## Experimental Results & Benchmarking

All experiments were executed on an NVIDIA A100-PCIE-40GB GPU using CUDA 12.1, mixed-precision `bfloat16`, a batch size of 8, and beam search decoding ($k=2$). We benchmarked three distinct setups: an untuned Zero-Shot baseline, the custom PyTorch LoRA model ($r=8, \alpha=32$), and a Full Fine-Tuning run. Both trainable configurations were optimised using AdamW (`train.py`), with learning rates set to $\text{LR} = 3 \times 10^{-4}$ for LoRA and $\text{LR} = 3 \times 10^{-5}$ for Full Fine-Tuning—reflecting the standard practice that low-rank adapter bottlenecks require an order-of-magnitude larger step size than unconstrained full-network updates to converge within a single epoch without destabilising pre-trained representations.

### Quantitative Performance & Resource Use

| Configuration | Trainable Params | Total Params | Train / Val Loss | Train Time (1 Epoch) | Peak Train VRAM | Peak Eval VRAM | Eval Throughput | `Val_Full` (R-1 / R-2 / R-L / R-Lsum) | `Val_Unseen` (R-1 / R-2 / R-L / R-Lsum) | Generalization Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Baseline** | 0 (0.00%) | 247.6M | N/A | N/A | N/A | 1.69 GB | 374.7 tok/s | 0.3493 / 0.1360 / 0.3118 / 0.3121 | 0.3663 / 0.1594 / 0.3188 / 0.3190 | +4.9% (no memorisation) |
| **Custom PyTorch LoRA** | **884,736 (0.36%)** | 248.5M | 1.0451 / 0.7289 | **~44 mins** | **6.59 GB** | **1.69 GB** | **307.8 tok/s** | **0.7066 / 0.5090 / 0.6477 / 0.6484** | **0.6792 / 0.4605 / 0.6087 / 0.6085** | **-3.9% (Target $\le 15\%$)** |
| **Full Fine-Tuning** | 247,577,856 (100.0%) | 247.6M | 0.8299 / 0.5678 | ~63 mins | 10.29 GB | 1.69 GB | 380.3 tok/s | 0.7317 / 0.5474 / 0.6780 / 0.6789 | 0.7040 / 0.4986 / 0.6389 / 0.6392 | -3.8% (Target $\le 15\%$) |

### Training Dynamics
![LoRA Loss Curve](lora_loss.png)
Figure 3: Training and Validation cross-entropy loss trajectories over $18,806$ steps on an NVIDIA A100 GPU.

Training converged smoothly over a single epoch, consisting of $18,806$ optimisation steps, using a batch size of $8$. Cross-entropy loss began at roughly $2.75$ on initial batches and steadily decreased to the final training loss of $1.0451$, with the validation loss reaching $0.7289$.

Notably, validation loss remained consistently lower than training loss throughout training. This can be attributed to active regularisation; during training passes, a dropout rate of $p=0.05$ is injected into the LoRA bottleneck to prevent over-reliance on specific token co-occurrences. During evaluation passes, the dropout is completely disabled, allowing the full deterministic capacity of the adapted network to be used without the injected noise. The active regularisation explains the difference between the losses.

The final validation loss of $0.7289$ corresponds to a perplexity of:

$$ e^{0.7289} \approx 2.07$$

Across the vocabulary of $32,128$ tokens, this indicates that the model is, on average, uncertain between barely two candidate tokens at each autoregressive decoding step.

### Takeaways

By restricting parameter updates strictly to the low-rank attention projections ($W_q$ and $W_v$), our custom LoRA implementation trained just $884{,}736$ parameters—a mere $0.36\%$ of the $248\text{M}$ parameter backbone. Despite updating less than $0.4\%$ of the network, LoRA captured $96.6\%$ of Full Fine-Tuning's ROUGE-1 ($0.7066$ vs $0.7317$) and $93.0\%$ of its ROUGE-2 ($0.5090$ vs $0.5474$). Crucially, freezing $99.64\%$ of the model eliminated the massive AdamW optimiser states required by full fine-tuning, cutting peak training VRAM from $10.29\text{ GB}$ down to $6.59\text{ GB}$. This $36\%$ reduction brings the entire training footprint comfortably below our $8.0\text{ GB}$ feasibility ceiling, proving that high-quality clinical adaptation is viable on consumer-tier hardware.

In comparison to the untuned Zero-Shot baseline, LoRA yielded dramatic relative gains of $+102.3\%$ in ROUGE-1 ($0.3493$ to $0.7066$) and $+274.3\%$ in ROUGE-2 ($0.1360$ to $0.5090$). This confirms that while generalist foundation models possess broad syntactic fluency, domain-specific sequence-to-sequence adaptation is indispensable for parsing dense radiological narratives into coherent lay summaries. Furthermore, evaluating across our deduplicated `Val_Unseen` cohort revealed a minimal performance drop of only $3.9\%$ in ROUGE-1 ($0.7066$ to $0.6792$) and $6.0\%$ in ROUGE-L ($0.6477$ to $0.6087$). Remaining well within our $\le 15\%$ acceptance hurdle verifies that the model learned genuine clinical translation semantics rather than over-indexing on repetitive normal screening templates.

## Clinical Safety

In biomedical natural language generation, standard lexical overlap metrics ($n$-gram ROUGE) serve as poor proxies for clinical safety. Because routine radiological reports frequently contain standard findings (e.g., "no pathologies present"), a generative model can attain deceptively high ROUGE scores by reciting fluent negative templates while hallucinating acute findings or omitting critical life-threatening pathologies. 

To evaluate factual clinical fidelity beyond surface statistics, four audits were conducted across validation cases covering anatomical lines, technical jargon simplification, acute airway placement, and layperson readability trade-offs:

| Case # & Focus | Source Report & Gold Standard | Custom LoRA Prediction ($r=8$) | Full Fine-Tuning Prediction | Clinical Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1**<br>*(Anatomy & Lines)* | **Source:** Central venous catheter traversing the left jugular vein with its tip in the superior vena cava. The remainder is unchanged.<br><br>**Gold:** A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before. | A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same. | A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else looks the same as before. | **Both Pass**<br>Anatomy and spatial coordinates preserved. No hallucinations. |
| **Case 2**<br>*(Jargon Simplification)* | **Source:** Bands of atelectasis and consolidation in the mid-lung fields, along with confluent opacities in the lower mid-lung fields, also showing bilateral involvement. Findings, considering the reported history, could correspond to COVID-19-type pneumonic infiltrates. Evaluate progression and correlate with clinical and laboratory data.<br><br>**Gold:** There are areas of lung collapse and thickening in the middle parts of the lungs, along with large blurry spots in the lower middle areas, affecting both lungs... | There are areas of lung collapse and lung thickening in the middle parts of the lungs, along with some cloudy areas in the lower parts of the lungs, which could be due to COVID-19-type pneumonia. It's important to consider the patient's history and laboratory results. | There are areas of lung collapse and solidified lung tissue in the middle and lower parts of the lungs, along with a mix of cloudy areas in the lower parts of the lungs, which are also seen on both sides... It's important to consider the patient's symptoms and lab tests to see if these changes are getting worse. | **Both Pass**<br>Jargon successfully simplified into plain terms; clinical instructions kept intact. |
| **Case 3**<br>*(Airway Safety)* | **Source:** Extensive consolidation in the right hemithorax. Left pleural effusion. Endotracheal tube with the distal tip positioned at the level of the right main bronchus.<br><br>**Gold:** There is a large area of lung tissue that has become solidified on the right side of the chest. There is also fluid buildup around the left lung. The breathing tube is in place, with its tip located at the level of the main airway on the right side. | There is a large area of lung thickening on the right side of the chest. There is fluid buildup on the left side of the chest. A breathing tube is positioned at the level of the right main airway. | There is a large area of lung thickening on the right side of the chest. There is fluid buildup around the left lung. The breathing tube is **correctly positioned** at the level of the main airway on the right side. | **LoRA: Pass** (neutral, faithful to source)<br>**Full FT: Critical Failure** (hallucinated *"correctly positioned"* on tube malposition) |
| **Case 4**<br>*(Readability vs. Hallucination)* | **Source:** Bilateral pulmonary involvement in the form of patchy areas of ground-glass opacity and scattered consolidation throughout both lungs, predominantly peripheral, consistent with COVID-19 pneumonia with progression of involvement as of 2 2 3 2 3. No pleural effusion, lymphadenopathy, or other relevant abnormalities are observed.<br><br>**Gold:** Both lungs are affected with irregular areas of haziness and scattered solid patches, mainly around the edges, which is typical for COVID-19 pneumonia and seems to be getting worse. There is no fluid around the lungs, swollen lymph nodes, or other noticeable issues. | Both lungs show signs of involvement in the form of **patchy areas of ground-glass opacity and scattered consolidation**, mainly at the edges, which is consistent with COVID-19 pneumonia, with the progression of involvement **as of 2 2 3 2 3**. There is no fluid around the lungs, lymph nodes, or other relevant abnormalities observed. | Both lungs show signs of involvement in the form of scattered areas of hazy opacity and scattered consolidation throughout both lungs, mainly at the edges, which is consistent with COVID-19 pneumonia and has gotten worse **since February 2nd, 3rd, and 3rd**. There is no fluid around the lungs, lymph nodes, or other significant issues seen. | **LoRA: Safe, Lower Readability** (retained raw jargon & OCR artifact)<br>**Full FT: Hallucination** (simplified jargon, but confabulated dates *"February 2nd, 3rd, and 3rd"*) |

### Interpretation of Case Study

The results of the audit show a characteristic vulnerability of language models within a clinical context. Full Fine-Tuning achieved higher ROUGE scores ($0.7317$ vs. $0.7066$), but produced a fatal medical hallucination. In Case 3, the Full FT model inserted "correctly positioned" into the output.

Updating all of the $248\text{M}$ parameters meant that FFT was prone to memorising common templates instead of sticking to the source text. Many of the reports within the training data described properly placed lines, which meant that the unconstrained model defaulted to "correctly positioned", potentially conveying dangerous misinformation to a patient.

In contrast, the low-rank bottleneck of LoRA prevents hallucinations by acting like a built-in regulariser. By freezing the base network and only tuning $W_q$ and $W_v$, LoRA was able to stay grounded within the encoder's hidden representations. There were no detected instances of dangerous clinical hallucinations.

However, Case 4 highlights the key trade-off of this regularisation: layperson readability versus factual safety. Because LoRA restricts updates to $0.36\%$ of the network, it acts conservatively when encountering noisy or unfamiliar inputs—retaining the technical phrase *"ground-glass opacity"* and copying the raw de-identification artifact `"as of 2 2 3 2 3"` verbatim rather than guessing its meaning. Conversely, while Full Fine-Tuning successfully translated *"ground-glass opacity"* into *"hazy opacity"*, it over-extrapolated on the noisy numbers `"2 2 3 2 3"` and hallucinated a completely fabricated clinical timeline (*"since February 2nd, 3rd, and 3rd"*). In a safety-critical patient portal, LoRA's conservative literalism is strongly preferable to Full Fine-Tuning's fluent confabulation.

## Recommended Model Deployment

In real-world deployment, the prevailing recommendation would be deploying the **FLAN-T5 base model with the custom LoRA ($r = 8$)** over the Full Fine-Tuning checkpoint.

While the FFT attained higher ROUGE scores ($+0.025$), LoRA captures over $96\%$ of the lexical performance, with a staggering $36\%$ less VRAM ($6.59\text{ GB}$ vs $10.29\text{ GB}$). The previous point is an aside, because the main consideration was ensuring that the output was clinically accurate, without producing any erroneous hallucinations that are not based on the data. The low-rank bottleneck of LoRA acted as a regulariser against template memorisation. 

However, there is a very important caveat to the deployment. Generative language models remain non-deterministic, so no sequence-to-sequence model should be deployed completely unmonitored to patients. It is for this reason that the below framework is recommended for real-world deployment:

```mermaid
flowchart TD
    Source(["Technical Radiology Report\n(Clinical Source Narrative)"])

    subgraph Tier1["Tier 1: Parameter-Efficient Neural Generation"]
        LoRA["FLAN-T5-Base + Custom LoRA (r = 8, α = 32)\nDeterministic Beam Search (k = 2)"]
        Draft["Candidate Plain-Language Summary\n+ Token Log-Probabilities"]
        LoRA --> Draft
    end

    subgraph RefPipe["Deterministic Clinical Reference Extraction"]
        Parser["Biomedical Entity & Acuity Parser\n(UMLS / RadGraph)"]
        RefState["Extracted Source Entities (E_source)\n& High-Acuity Pathology Flags"]
        Parser --> RefState
    end

    Source --> LoRA
    Source --> Parser

    subgraph Tier2["Tier 2: Automated Clinical Safety Verification"]
        G1["1. Entity Grounding Check\nVerify E_summary ⊆ E_source\n(Zero Hallucinations)"]
        G2["2. Acute Omission Watchlist\nVerify 0 Missed Critical Findings\n(e.g. Malposition, Pneumothorax)"]
        G3["3. Generation Confidence Filter\nVerify Minimum Token Probability\nτ ≥ 0.85"]
    end

    Draft --> G1
    Draft --> G2
    Draft --> G3
    RefState -.->|"Ground-Truth Entities"| G1
    RefState -.->|"Required Acute Entities"| G2

    Gate{"All 3 Clinical Safety\nGates Passed?"}

    G1 -->|"Grounded"| Gate
    G2 -->|"Complete"| Gate
    G3 -->|"Confident"| Gate

    Pass(["Release Verified Summary\nto Patient Portal"])
    Fail(["Withhold Summary & Escalate\nto Radiologist Review Queue"])

    Gate ==>|"PASS: Verified Safe"| Pass
    Gate -.->|"FAIL: Safety Trigger"| Fail

    classDef inputNode fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef tier1Node fill:#ede9fe,stroke:#6d28d9,stroke-width:2px,color:#2e1065
    classDef parserNode fill:#ccfbf1,stroke:#0d9488,stroke-width:2px,color:#115e59
    classDef guardNode fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef gateNode fill:#f8fafc,stroke:#334155,stroke-width:2.5px,color:#0f172a
    classDef passNode fill:#dcfce7,stroke:#15803d,stroke-width:2.5px,color:#14532d
    classDef failNode fill:#fee2e2,stroke:#b91c1c,stroke-width:2.5px,color:#7f1d1d

    class Source inputNode
    class LoRA,Draft tier1Node
    class Parser,RefState parserNode
    class G1,G2,G3 guardNode
    class Gate gateNode
    class Pass passNode
    class Fail failNode

    style Tier1 fill:#f5f3ff,stroke:#8b5cf6,stroke-width:1.5px,stroke-dasharray: 5 5,color:#4c1d95
    style RefPipe fill:#f0fdfa,stroke:#14b8a6,stroke-width:1.5px,stroke-dasharray: 5 5,color:#134e4a
    style Tier2 fill:#fffbeb,stroke:#f59e0b,stroke-width:2px,color:#78350f
```

### Two-Tier Guardrail Architecture

Tier 1 generates candidate summaries using `google/flan-t5-base` with custom LoRA ($r=8, \alpha=32$) under deterministic beam search ($k=2$, length penalty $1.0$). Stochastic sampling is disabled entirely to guarantee consistent, repeatable clinical translations. Before reaching the patient portal, Tier 2 intercepts each draft through three automated checks: an entity grounding filter (RadGraph/UMLS) enforcing $E_{\text{summary}} \subseteq E_{\text{source}}$ to block hallucinations, an acute watchlist preventing omissions of critical pathologies (like pneumothoraces or tube malpositions), and a confidence gate requiring minimum token probabilities of $\tau \ge 0.85$. Summaries clearing all three gates release automatically; any flagged case routes directly to an attending radiologist for review.

### Operational & Hardware Sizing

Inference requires just $1.69\text{ GB}$ of VRAM, allowing deployment on budget cloud GPUs (NVIDIA T4) or standard consumer hardware (RTX 3060/4060) without multi-GPU overhead. Generating at $307.8\text{ tokens/second}$ yields an average latency under $0.25\text{ seconds}$ per report, easily supporting real-time hospital portal workloads.

## Dependencies & Execution Guide

All experiments were implemented using PyTorch modules, without any high-level library wrappers.

### Dependencies & Requirements

| Component | Minimum Version & Role |
| :--- | :--- |
| **Python / CUDA** | Python $\ge 3.10$ with CUDA 12.1 (`bfloat16` support) |
| **PyTorch (`torch`)** | $\ge 2.1.0$ — custom LoRA modules, loss masking, and DDP |
| **Hugging Face (`transformers`, `datasets`)** | $\ge 4.38.0$ / $\ge 2.18.0$ — `flan-t5-base` weights and `LaymanRRG` data |
| **Evaluation (`evaluate`, `rouge-score`)** | $\ge 0.4.1$ / $\ge 0.1.2$ — ROUGE-1, ROUGE-2, and ROUGE-L metrics |
| **Utilities (`sentencepiece`, `matplotlib`)** | $\ge 0.2.0$ / $\ge 3.8.0$ — T5 tokeniser backend and loss plotting |

#### Automated Installation Script

Below is a script that creates a virtual environment and installs the dependencies needed to run the project.

```bash
conda create -n clinical-flan python=3.10 -y
conda activate clinical-flan
pip install torch transformers datasets evaluate rouge-score sentencepiece matplotlib
```

### Reproducibility & CLI Execution

All pipelines execute directly from `recognition/FLAN-T5-RRG-s4436813/`:

| Pipeline Stage | CLI Command | Generated Artifacts & Telemetry |
| :--- | :--- | :--- |
| **1. Zero-Shot Eval** | `python predict.py --mode zero_shot --num_samples 500` | Baseline ROUGE (`Val_Full` & `Val_Unseen`), VRAM & throughput |
| **2. Train LoRA (1 GPU)** | `python train.py --mode lora` | Saves `checkpoints/lora.pt` and plots `lora_loss.png` |
| **3. Train LoRA (DDP)** | `torchrun --nproc_per_node=2 train.py --mode lora` | Multi-GPU Distributed Data Parallel training |
| **4. Train Full FT** | `python train.py --mode full_finetune` | Saves `checkpoints/full_finetune.pt` ($100\%$ weights) |
| **5. Evaluate LoRA** | `python predict.py --mode lora --checkpoint checkpoints/lora.pt --num_samples 500` | LoRA ROUGE scores, latency telemetry & 4-case safety autopsy |
| **6. Evaluate Full FT** | `python predict.py --mode full_finetune --checkpoint checkpoints/full_finetune.pt --num_samples 500` | Full FT ROUGE scores & comparative safety autopsy |

## Artificial Intelligence Usage Disclosure

In compliance with the COMP3710 Assessment Guidelines, this section documents the nature, scope, and verification of Generative Artificial Intelligence tools utilised during this project:

* **AI Tools Utilised:** Google Gemini (accessed via the Antigravity agentic development environment).
* **Specific Tasks Assisted:**
  * *Visualisation & Plotting:* Assisting in styling the Mermaid deployment flowchart and scaffolding the Matplotlib layout script for the training loss trajectory (`lora_loss.png`).
  * *HPC Scripting Templates:* Drafting boilerplate Slurm batch submission scripts (`sbatch`) for scheduling interactive and distributed GPU runs on the UQ Rangpur HPC cluster.
  * *Report Review & Formatting:* Proofreading report sections for clarity, formatting KaTeX mathematical notation, and checking compliance against the course marking rubric.
* **Auditing, Verification & Empirical Testing:**
  * *Mathematical Auditing:* Manually derived the Low-Rank Adaptation formulation ($h = W_0 x + \frac{\alpha}{r} B A x$) and verified the initialization scheme ($B = 0$, $A \sim \mathcal{N}(0, \sigma^2)$) to ensure that the adapter evaluates to zero perturbation ($\Delta W = 0$) at step 0 without corrupting base model representations.
  * *Data Pipeline Verification:* Audited the data collation logic in PyTorch to confirm target padding tokens were strictly mapped from T5's default (`0`) to `-100` for cross-entropy loss masking, and validated the SHA-256 report hashing algorithm to ensure zero template leakage into `Val_Unseen`.
  * *Empirical Benchmarking:* Validated all scripts with `py_compile` and empirically tested single-GPU and distributed `torchrun` execution directly on NVIDIA A100 nodes, actively tracking step-level loss convergence, peak VRAM allocations, and inference throughput.

**Verification Responsibility:** In accordance with course policy, the author, Aidan Fassy (s4436813), assumes 100% personal responsibility for every line of code, loss formula, evaluation metric, and clinical interpretation presented in this submission.