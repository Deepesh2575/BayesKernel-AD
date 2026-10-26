# BayesKernel-AD

BayesKernel-AD is an educational hybrid for multivariate anomaly detection on
streaming data. It combines Gaussian-mixture likelihoods, fuzzy c-means
memberships, a soft-margin kernel classifier, and minimum expected-loss
decisions. The implementation emphasizes transparent syllabus mapping rather
than claiming production-grade performance.

This project is licensed under the [MIT License](LICENSE).

## Syllabus mapping

| Unit | Implementation |
| --- | --- |
| I – Bayesian decision theory | Class-conditional posteriors, log-opinion pooling, configurable cost matrix, expected risk, and uncertainty in `models/decision.py` |
| II – Estimation and model selection | Gaussian/GMM MLE via EM, regularized empirical-Bayes MAP approximation, AIC/BIC selection, and mutual information in `models/` |
| III/V – Kernel methods | Linear, RBF, polynomial, and GMM Fisher kernels with soft-margin SVM probabilities in `models/kernel_classifier.py` |
| IV – Adaptive clustering | Missing-value-capable fuzzy c-means, fuzzy-initialized EM clustering, and a streaming feature-shift detector in `models/clustering.py` |

## Setup

Use Python 3.10 or newer. In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py demo
python main.py demo --no-plot
python main.py evaluate
python -m unittest discover -s tests -v
```

The demo trains on a labeled warm-up window, then predicts individual samples
as the stream continues. It reports posterior anomaly probabilities and drift
adaptation delay. `evaluate` compares the hybrid against pure generative,
discriminative, and fuzzy-clustering baselines on a synthetic dataset and
scikit-learn's bundled Wisconsin breast-cancer dataset. The latter is a binary
classification proxy (malignant cases are class 1), not an unsupervised
anomaly benchmark. Results use identical stratified holdouts within each
dataset; because the real dataset is not temporal, it does not measure drift.
The synthetic hybrid row reports drift-detection delay in samples; the
non-adaptive baselines and non-temporal real dataset report this metric as
not applicable. Soft classification accuracy is the mean probability assigned
to the true class, while expected risk is the observed average cost-matrix loss.

## Design and limitations

* Labels use `0 = normal` and `1 = anomaly`. The warm-up set must contain at
  least two examples of each class.
* The generative MAP method is explicitly an empirical-Bayes shrinkage
  approximation after EM, not a full Bayesian posterior inference procedure.
* The Fisher kernel maps GMM responsibilities and standardized component
  residuals to a feature vector, then uses its inner product as a positive
  semidefinite kernel.
* The drift detector compares normalized means in a rolling window. On drift,
  the pipeline refits on retained labeled history. Unlabeled data can trigger
  detection but cannot update supervised calibration; call `partial_fit` with
  labels when newly labeled observations become available.
* Kernel probability outputs use scikit-learn's internal probability
  calibration. If deployment calibration matters, use a time-ordered
  calibration set and monitor calibration drift.
* `models/model_selection.py` provides joint AIC/BIC-rank mixture selection,
  mutual-information fuzzy-cluster selection, kernel cross-validation, and a
  BIC-based update criterion. Set `PipelineConfig(n_components=None,
  n_clusters=None, kernel="auto")` to select these automatically. On drift,
  high-confidence pseudo-labels from the detected window seed an adaptive
  refit; labeled `partial_fit` updates are gated by BIC improvement.
* `GaussianRegimeModel.fit_maximum_entropy` fits the Gaussian distribution that
  maximizes differential entropy subject to empirical first- and second-moment
  constraints.

The key implementation modules are `core/pipeline.py`, `models/generative.py`,
`models/clustering.py`, `models/kernel_classifier.py`, and
`models/decision.py`.
