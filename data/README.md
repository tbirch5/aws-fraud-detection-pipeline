This folder mirrors the S3 data layer used in the pipeline:

- raw/ → incoming datasets
- processed/ → transformed features used for training
- test/ → historical combined test set used for drift detection

Note: In production, these are stored in S3 and persist across runs to simulate real-world data accumulation.