# CODA

CODA is a content verification engine designed to help users evaluate the credibility of digital information shared through links or online sources.

The idea behind CODA is to address a growing problem in the digital space — misleading, emotionally manipulative, or unverified content that spreads quickly without context. By combining natural language processing with source-based verification, CODA aims to offer a more informed way to examine online content before trusting or sharing it.

## What CODA does

A user provides a link to an article or online content. Once submitted, CODA processes the textual content and performs linguistic analysis using NLP techniques to examine patterns such as tone, framing, emotional intensity, and manipulative language indicators.

Based on this analysis, the system generates an evaluation score that reflects how likely the content is to contain manipulative or misleading language.

Alongside linguistic analysis, CODA also checks whether the same topic or claim appears across multiple news sources, helping users compare information and identify whether the content is supported by broader reporting.

## Core Functions

* Link-based content input
* NLP-based linguistic analysis
* Detection of manipulative language patterns
* Credibility / influence scoring
* Cross-verification using multiple news sources

## Why this project was built

CODA was developed as an attempt to explore how language itself can influence perception, especially in online environments where information moves quickly and often without verification.

Rather than simply labeling content as true or false, the goal is to help users understand how content is written, how it may affect interpretation, and whether it aligns with information available from other sources.

## Tech Stack

* Python
* Flask
* NLP modules
* HTML / CSS
* Integrated model-based processing

## Project Structure

* `app.py` – main application logic
* `model/` – scoring / processing models
* `nlp/` – linguistic analysis components
* `templates/` – frontend pages
* `static/` – styles and assets
* `uploads/` – temporary input handling

## Developed by

Muskaan Hameed
