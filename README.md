The Social Media Extremism Detection Challenge invites participants to develop machine learning models that distinguish potentially extremist content from regular social media posts. This competition is part of a Community Impact Initiative focused on machine learning for social awareness and social good.

# Social Media Extremism Detection

Binary classification of extremist social media posts using transformer models.

**Course:** Advanced Machine Learning (F1801Q151) | Ruben Vandamme & Muhamad Khatib

## Task

Classify anonymized English social media posts as extremist or not. The dataset contains 2,250 labeled training samples from a hate speech detection corpus, hand-labeled based on whether posts promote violent ideologies or support extremist organizations.

## Approach

BERT-style transformer with a small linear classifier head, trained with BCE loss and AdamW. Encoder was initially frozen then gradually unfrozen during training.

## Experiments

- **Model selection:** Compared BERT, RoBERTa, HateBERT, and BERTweet. HateBERT underperformed despite being domain-specific; RoBERTa and BERTweet generalized better.
- **Pooling:** Mean pooling over the last few layers slightly beat CLS-token pooling.
- **Data augmentation:** Back translation and entity swapping underperformed simpler methods like contextual word substitution and random insertion/swap/deletion.
- **Cross-validation:** 5-fold CV with soft voting gave consistent gains.
- **Class weighting:** Upweighted non-extremist examples to account for test set imbalance.

## Results

Best accuracy: **77%**. Naively combining individually well-performing components dropped performance to ~73%, showing that changes interact in non-obvious ways. Estimated ceiling with better augmentation is around 80%.

## Main Takeaways

- Picking the right base model matters more than tuning the classifier head
- Validation loss is a more stable training signal than accuracy
- Simple, diverse augmentations beat targeted semantic ones
- Optimization is non-linear: improvements don't stack additively
