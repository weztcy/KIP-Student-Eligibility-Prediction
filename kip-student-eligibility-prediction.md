# Portfolio Project Data — KIP Student Eligibility Prediction

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/kip_student_eligibility_prediction/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image11.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "kip-student-eligibility-prediction",
  "code": "PRO-15 // ML-02",
  "title": "KIP Student Eligibility Prediction",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/KIP-Student-Eligibility-Prediction",
  "license": "MIT License",
  "banner": "/images/projects/kip_student_eligibility_prediction/banner.webp",
  "summary": "Gaussian Naive Bayes binary classifier and Streamlit decision-support app predicting KIP educational aid eligibility from student records.",
  "projectDescription": "Built a supervised Gaussian Naive Bayes classification pipeline and Streamlit inference web application using 500 socioeconomic student records from SMP Sepuluh November 02 Semarang to predict Kartu Indonesia Pintar (KIP) eligibility, complete with probability calibration diagnostics and visual EDA.",
  "architectureSummary": "Tabular CSV Ingestion -> One-Hot Feature Encoding -> GaussianNB Training -> Pickle Serialization -> Diagnostic Evaluation & Streamlit Web Inference",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "Streamlit",
    "Matplotlib",
    "Seaborn",
    "Pillow",
    "Jupyter Notebook"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Binary Classification",
    "targetPlatform": "Streamlit Web & Python Runtime",
    "primaryDomain": "Educational Aid Screening",
    "architecturePipeline": "Encoding -> GaussianNB -> Web Serving",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Tabular Classification & Web Serving",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Socioeconomic Educational Aid Screening",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "Binary Classification",
      "Threshold Diagnostics",
      "Streamlit Inference"
    ]
  },
  "mlProblemContext": "Screening junior high school applicants for Kartu Indonesia Pintar (KIP) scholarships requires balancing household income, family size, commuting constraints, and prior aid records. This project automates eligibility classification across 500 student records while diagnosing severe class-imbalance effects under probabilistic modeling.",
  "mlDatasetOverview": {
    "profileBadge": "500 Student Records",
    "source": "SMP Sepuluh November 02 Semarang student roster (dataset.csv)",
    "modality": "Tabular",
    "scale": "500 rows, 7 raw attributes (0 missing values)",
    "inputShape": "500 x 10 Encoded Features",
    "dataFormat": "CSV",
    "description": "Private school socioeconomic dataset containing parental income, KPS status, transportation mode, sibling count, school distance, BSM recommendation, and KIP recipient labels (408 Tidak, 92 Ya).",
    "provenanceNote": "Derived from school administrative records (daftar_pd-SMP SEPULUH NOPEMBER 2) and bundled as dataset.csv under MIT License."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "10D Mixed Feature Space",
    "spaceNote": "3 numeric features combined with 7 one-hot encoded indicators across 3 categorical fields.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "penerima_kip",
      "type": "Binary Class",
      "distribution": "408 Non-Recipients (81.6%) vs 92 KIP Recipients (18.4%)"
    },
    "inputSignals": [
      {
        "name": "rekomendasi_bsm",
        "type": "Categorical",
        "importanceOrRole": "Primary Positive Signal (r = +0.21)",
        "description": "Prior BSM scholarship recommendation; 65.2% of KIP recipients hold BSM recommendations versus 38.0% of non-recipients."
      },
      {
        "name": "penghasilan",
        "type": "Numerical",
        "importanceOrRole": "Core Socioeconomic Scale (IDR 0.8M–4.0M)",
        "description": "Monthly parental income in IDR; exhibits strong collinearity with transport mode (r = +0.79 private vehicle, r = -0.83 walking)."
      },
      {
        "name": "peserta_kps",
        "type": "Categorical",
        "importanceOrRole": "Social Safety Net Indicator",
        "description": "Family Social Protection Card (KPS) ownership indicator (`Ya` / `Tidak`), negatively correlated with parental income (r = -0.68)."
      },
      {
        "name": "alat_transportasi",
        "type": "Categorical",
        "importanceOrRole": "Commuting Proxy (3 Categories)",
        "description": "Student school transport mode (`Jalan kaki`, `Angkutan umum`, `Kendaraan pribadi`), reflecting household mobility tiers."
      },
      {
        "name": "jumlah_saudara_kandung",
        "type": "Numerical",
        "importanceOrRole": "Household Dependency Factor (1–3)",
        "description": "Number of siblings in the household; students with 3 siblings show a 28.1% KIP recipient rate."
      },
      {
        "name": "jarak_rumah_ke_sekolah",
        "type": "Numerical",
        "importanceOrRole": "Accessibility Metric (1–5 Scale)",
        "description": "Ordinal/kilometer home-to-school distance scale ranging from 1 to 5 across student households."
      }
    ],
    "dataCharacteristicsSummary": "Exploratory correlation and multivariate scatter analyses reveal strong internal structure among socioeconomic predictors—income strongly separates transport tiers—yet heavy distribution overlap between KIP and non-KIP classes, with BSM recommendation emerging as the clearest univariate differentiator."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Learning — Probabilistic Binary Classification",
    "methodologyDescription": "Formulated KIP eligibility as supervised binary classification using Gaussian Naive Bayes (GaussianNB) over one-hot encoded tabular attributes. Combined closed-form likelihood estimation with post-hoc Youden's Index ROC threshold analysis to diagnose minority-class sensitivity under class imbalance.",
    "justification": "Gaussian Naive Bayes delivers fast, deterministic, low-variance training on small 500-row datasets and exposes interpretable class-conditional probability outputs directly suitable for transparent Streamlit screening."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Dataset Ingestion & Target Isolation",
      "description": "Loaded 500 clean records from dataset.csv and separated the six raw socioeconomic predictors from the binary penerima_kip target column.",
      "badge": "CSV Load"
    },
    {
      "step": "02",
      "title": "Categorical One-Hot Encoding",
      "description": "Expanded peserta_kps, alat_transportasi, and rekomendasi_bsm via pd.get_dummies into seven integer-cast binary columns alongside three numeric predictors.",
      "badge": "One-Hot"
    },
    {
      "step": "03",
      "title": "Binary Label Encoding",
      "description": "Mapped target categories deterministically using {'Tidak': 0, 'Ya': 1} to align with Scikit-learn binary probability indexing.",
      "badge": "Label Map"
    },
    {
      "step": "04",
      "title": "Serving Schema Alignment",
      "description": "Concatenated live Streamlit sidebar or uploaded CSV records onto dataset.csv prior to pd.get_dummies, slicing df[:1] to guarantee all 10 feature columns.",
      "badge": "Schema Lock"
    }
  ],
  "mlDataPreparationNote": "Training in Model Klasifikasi KIP.ipynb fit on all 500 records without a holdout split; inference concatenates dataset.csv exclusively to preserve identical dummy-column ordering.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Gaussian Naive Bayes (GaussianNB)",
    "architectureSummary": "Single-stage probabilistic generative classifier modeling per-class feature likelihoods as independent Gaussian distributions multiplied by empirical class priors [0.816, 0.184].",
    "configurationBadge": "Closed-Form Fit",
    "hyperparameters": [
      {
        "name": "priors",
        "value": "None (Empirical [0.816, 0.184])",
        "description": "Computes class prior probabilities directly from observed training class counts (408 Tidak, 92 Ya)."
      },
      {
        "name": "var_smoothing",
        "value": "1e-09",
        "description": "Adds a stability fraction of the largest feature variance to all per-feature variances."
      }
    ],
    "objectiveCriterion": "Maximum A Posteriori (Gaussian Likelihood)",
    "optimizationSolver": "Closed-Form Mean & Variance Estimation",
    "trainingProcess": "Instantiated GaussianNB with default smoothing and fitted across all 500 encoded student vectors in a single closed-form pass inside Model Klasifikasi KIP.ipynb. Because parameter estimation computes exact class means and variances analytically, training is deterministic and serialized directly to modelNBC_kip.pkl."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Full-Dataset Diagnostic & ROC Threshold Audit",
    "description": "Evaluated the serialized modelNBC_kip.pkl across all 500 records in generate_figures.py using default 0.50 cutoff, Youden's Index ROC threshold optimization (0.1811), ROC-AUC, and Precision-Recall curves.",
    "metrics": [
      {
        "label": "DEFAULT ACCURACY",
        "value": "81.60%",
        "benchmark": "Target > 85.00%"
      },
      {
        "label": "ROC AUC SCORE",
        "value": "0.522",
        "benchmark": "Baseline > 0.500"
      },
      {
        "label": "PR AVERAGE PRECISION",
        "value": "0.196",
        "benchmark": "Baseline > 0.184"
      },
      {
        "label": "TUNED SENSITIVITY",
        "value": "55.43%",
        "benchmark": "Threshold >= 50.00%"
      },
      {
        "label": "TUNED SPECIFICITY",
        "value": "51.72%",
        "benchmark": "Threshold >= 50.00%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit KIP Prediction Interface",
      "badge": "Web UI",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image01.webp",
      "telemetryNote": "Interactive sidebar sliders, CSV upload, encoded 10-feature preview, and real-time classification output."
    },
    {
      "title": "Target Class Distribution",
      "badge": "Class Balance",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image02.webp",
      "telemetryNote": "408 Non-Recipients (81.6%) vs 92 KIP Recipients (18.4%) across 500 records."
    },
    {
      "title": "Parental Income Distribution by KIP",
      "badge": "KDE & Boxplot",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image03.webp",
      "telemetryNote": "Mean income IDR 2,438,725 (Tidak) vs IDR 2,393,478 (Ya) across IDR 0.8M–4.0M."
    },
    {
      "title": "Prior Aid Impact (KPS & BSM)",
      "badge": "Stacked Bar",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image04.webp",
      "telemetryNote": "BSM recommendation lifts KIP recipient share from 11.2% (n=32) to 27.9% (n=60)."
    },
    {
      "title": "Transportation Mode & Acceptance",
      "badge": "Grouped Bar",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image05.webp",
      "telemetryNote": "KIP rates across walking (18.4%), public transit (19.3%), and private vehicle (17.4%)."
    },
    {
      "title": "Siblings & School Distance Analysis",
      "badge": "Stacked Bar",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image06.webp",
      "telemetryNote": "3-sibling households exhibit 28.1% KIP eligibility versus 14.1%–18.5% for 1–2 siblings."
    },
    {
      "title": "Pearson Feature Correlation Heatmap",
      "badge": "Correlation Matrix",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image07.webp",
      "telemetryNote": "rekomendasi_bsm_Ya leads target correlation (+0.21); income strongly correlates with transport."
    },
    {
      "title": "Confusion Matrix Threshold Comparison",
      "badge": "Confusion Matrix",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image08.webp",
      "telemetryNote": "Default 0.50 collapses to majority class (0 TP); optimal 0.181 threshold recovers 51 TP."
    },
    {
      "title": "ROC & Precision-Recall Curves",
      "badge": "Discrimination Curve",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image09.webp",
      "telemetryNote": "ROC-AUC = 0.522 and PR-AP = 0.196 against 0.184 random prevalence baseline."
    },
    {
      "title": "Multivariate Income vs Distance Scatter",
      "badge": "Multivariate Plot",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image10.webp",
      "telemetryNote": "Clear income clustering by transport mode while KIP recipients span income bands."
    },
    {
      "title": "Recipient vs Non-Recipient Profile",
      "badge": "Profile Comparison",
      "imageUrl": "/images/projects/kip_student_eligibility_prediction/image11.webp",
      "telemetryNote": "BSM recommendation gap (65.2% vs 38.0%) stands out over nearly equal feature means."
    }
  ],
  "mlVisualizationTelemetryNote": "All 11 visualizations verify dataset distributions and demonstrate how uncalibrated 0.50 probability cutoffs underperform compared to Youden-tuned thresholds on imbalanced tabular data.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Model likelihoods are governed by empirical class priors (81.6% Tidak) and unscaled parental income variance, causing default predictions to favor the majority class. Post-hoc threshold tuning to 0.1811 restores minority detection, while EDA confirms BSM recommendation as the strongest eligibility driver.",
    "keyDrivers": [
      {
        "factor": "Prior BSM Recommendation (rekomendasi_bsm)",
        "impact": "Strongest positive correlation with KIP status (r = +0.21); 65.2% prevalence among recipients vs 38.0% non-recipients."
      },
      {
        "factor": "Unscaled Income Variance & Class Prior",
        "impact": "Large IDR variance (10^6 scale) alongside 0.816 negative prior pushes posterior probabilities below 0.50 without threshold calibration."
      },
      {
        "factor": "Household Sibling Count (3 Siblings)",
        "impact": "Students with 3 siblings show a 28.1% KIP recipient rate compared to 14.1%–18.5% for smaller households."
      }
    ],
    "limitations": [
      {
        "title": "Absence of Train-Test Holdout Split",
        "description": "Model Klasifikasi KIP.ipynb fits GaussianNB on all 500 records, preventing out-of-sample generalization measurement in the current artifact."
      },
      {
        "title": "Default Threshold Minority Collapse",
        "description": "At the standard 0.50 cutoff used in kip_webapp.py, all 92 positive KIP cases are classified as Tidak due to class imbalance and unscaled features."
      },
      {
        "title": "Overlapping Feature Distributions",
        "description": "Parental income, distance, and transport distributions overlap heavily between classes, capping GaussianNB ROC-AUC at 0.522."
      }
    ],
    "futureImprovements": [
      {
        "title": "Stratified Cross-Validation & Feature Scaling",
        "target": "Evaluate 5-fold stratified CV with StandardScaler/MinMaxScaler and SMOTE oversampling."
      },
      {
        "title": "Tree-Based Ensemble Benchmarking",
        "target": "Benchmark Random Forest, XGBoost, and Logistic Regression with class weights to improve ROC-AUC above 0.75."
      },
      {
        "title": "Calibrated Threshold Integration in Streamlit",
        "target": "Expose tunable probability cutoffs (e.g., 0.181) and risk tiers inside streamlit/kip_webapp.py."
      }
    ],
    "discoveryNote": "Diagnostic audit highlights the importance of inspecting confusion matrices and PR curves beyond headline accuracy on imbalanced school-aid datasets."
  },
  "mlModelArtifacts": [
    {
      "name": "modelNBC_kip.pkl",
      "format": "PKL"
    },
    {
      "name": "streamlit/modelNBC_kip.pkl",
      "format": "PKL"
    },
    {
      "name": "dataset.csv",
      "format": "CSV"
    },
    {
      "name": "1.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Binary Classification",
    "modelArchitecture": "Gaussian Naive Bayes",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Full-Dataset ROC Audit"
  }
}
```
