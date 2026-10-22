# Translating Radiology Findings into Plain Language: Evaluating FLAN-T5 and Custom PyTorch LoRA on LaymanRRG

## Aidan Fassy – 44368133

### Executive Summary

Within healthcare and medicine, radiological reports have the reputation of being extremely technical, filled with extensive medical jargon and very unfriendly for an unskilled person to read. As a result, patients who desire transparency when it comes to their healthcare, are often left anxious, confused and prone to misinterpretations. It stands that there is a need for a layperson to understand and interpret their report correctly.

It is for this reason that the objective of this project is to build a sequence-to-sequence NLP prototype that translates jargon-heavy complex diagnostic findings into plain, accurate and simple layperson accessible language – ideally at the 6th grade level. This project uses the ACL 2025 BioLaySum (LaymanRRG) open-source benchmark dataset.

The google/flan-t5-base model (~248M parameters) is used, using a custom PyTorch Low-Rank Adaptation (LoRA) layer, which is injected into the attention query ($W_q$) and the value ($W_v$) projections. 