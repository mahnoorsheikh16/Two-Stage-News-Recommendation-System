# Two-Stage News Recommendation via Candidate Generation and Ranking Models
This project builds and evaluates an industry-style two-stage recommender:
1. **Candidate Generation** narrows down the full news catalogue to a high-recall candidate pool using content similarity and behavioural priors.
2. **Ranking** applies increasingly expressive models to rank candidates and optimize top-K recommendation quality.
The system is designed to mirror real-world recommender pipelines, where recall quality in the first stage directly determines the ceiling for ranking performance.

*This is a course project for CSE 847 Machine Learning at Michigan State University.

## Table of Contents
1. [Dataset and Preprocessing](#dataset-and-preprocessing)
2. [Stage 1: Candidate Generation (Recall-Oriented)](#stage-1-candidate-generation-recall-oriented)
3. [Stage 2: Ranking Models (Precision-Oriented)](#stage-2-ranking-models-precision-oriented)
4. [Key Results](#key-results)

## Dataset and Preprocessing
[MIND-small (Microsoft News Dataset)](https://msnews.github.io/) contains ~65k news articles, ~230k user impression logs, user click histories, timestamps, categories, sources, and article metadata. Both training and validation splits are used, with strict chronological handling to prevent data leakage.

News Metadata Preprocessing
- HTML and noise removal (BeautifulSoup + regex)
- Standardization of punctuation and whitespace
- Title + abstract concatenation for robust content representation
- Identification of very short articles to exclude low-quality candidates
- Publisher/source reconstruction from URLs

User Behavior Preprocessing
- Expansion of impression logs into (user, article, label) format
- Construction of ordered user click histories
- Computation of smoothed CTR priors (news, category, source) using Laplace smoothing

## Stage 1: Candidate Generation (Recall-Oriented)
Candidate articles are scored using a weighted combination of:
- TF–IDF content similarity (Unigrams + bigrams, and Cosine similarity over recent clicked articles)
- Popularity priors (news-level CTR with fallbacks to category/source/global CTR)
- User category preferences (temperature-scaled softmax over historical clicks)
- Source familiarity bonus (previously clicked publishers)
The final candidate score is a weighted sum of these components. Candidate pool size K is selected via Recall@K, with K = 500 chosen as the best trade-off between recall and computation.

## Stage 2: Ranking Models (Precision-Oriented)
Each impression’s candidate pool is ranked using multiple models:

Baselines: Logistic Regression (TF–IDF), XGBoost (TF–IDF)
Neural Rankers: MLP v1: DistilBERT embeddings only, MLP v2 (Best Model)

Specifications for MLP v2 (Best Model):
- TF–IDF reduced via Truncated SVD
- Concatenated with DistilBERT embeddings
- Trained using hard negative sampling from the candidate pool
This hybrid model captures both lexical precision and semantic relevance, leading to large gains in personalised ranking.

## Key Results
All evaluation is performed at the impression level using Hit@K, MRR@K, nDCG@K. Both two-stage and one-stage (standard MIND setup) evaluations are reported for comparison.

Candidate generation achieves Recall@500 ≈ 0.52, setting a strong recall ceiling.

Traditional models perform reasonably in one-stage ranking but struggle in two-stage settings.

Hybrid MLP (TF–IDF + BERT) delivers ~10× improvement in Hit@5, and large gains in MRR@5 and nDCG@5.

Results highlight the importance of high-quality candidate recall, hybrid feature representations, and hard-negative training aligned with the evaluation distribution.
