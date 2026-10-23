# Translating Radiology Findings into Plain Language: Evaluating FLAN-T5 and Custom PyTorch LoRA on LaymanRRG – Aidan Fassy – 44368133

## Executive Summary

Within healthcare and medicine, radiological reports have the reputation of being extremely technical, filled with extensive medical jargon and very unfriendly for an unskilled person to read. As a result, patients who desire transparency when it comes to their healthcare, are often left anxious, confused and prone to misinterpretations. It stands that there is a need for a layperson to understand and interpret their report correctly.

It is for this reason that the objective of this project is to build a sequence-to-sequence NLP prototype that translates jargon-heavy complex diagnostic findings into plain, accurate and simple layperson accessible language – ideally at the 6th grade level. This project uses the ACL 2025 BioLaySum (LaymanRRG) open-source benchmark dataset.

The google/flan-t5-base model (~248M parameters) is used, using a custom PyTorch Low-Rank Adaptation (LoRA) layer, which is injected into the attention query ($W_q$) and the value ($W_v$) projections. A goal of this investigation was also to avoid heavy parameter fine-tuning by freezing ~98.6% of the base weights and only updating a lightweight rank-8 bottleneck (~$884\text{K}$ parameters, $0.4\%$ of total).

However, a problem still remains: standard metrics like ROUGE-1, ROUGE-2 and ROUGE-L measure surface n-gram overlap, not the underlying clinical truth. As a result of this, a model can subvert the testing objectives by memorising normal radiological findings (e.g. "lungs are clear, no acute abnormality") while omitting crucial clinical details, some of these can be as severe as hallucinating non-existent tumors.

So the current investigation is to evaluate whether LoRA fine-tuning is able to genuinely improve factual clinical fidelity over an untuned zero-shot baseline, or if it simply increases superficial linguistic fluency on memorised screening templates across both full, and deduplicated splits.

## Architecture & Working Principles

- Explain Encoder-Decoder architecture, and why its used

**Still to do: Why is there an advantage over decoder only???**

A FLAN-T5 is fundamentally an *Encoder-Decoder* architecture, and subsequently has a huge architectural advantage over *decoder-only* causal models (e.g. GPT-2). Findings in radiological reports are scattered, non-linear and conditional, radiologists may mention an acute finding in a single sentence, describe normal anatomy for three, and then add crucial context in the last sentence – this means that each sentence isn't necessarily weighed equally.

Within FLAN-T5, a bidirectional encoderis used.


- Explain the PyTorch LoRA mechanism – get into the mathematics

- Explain choosing $r$ and $\alpha$ values.

- Insert a diagram (mermaid or other) into this section to explain the architecture