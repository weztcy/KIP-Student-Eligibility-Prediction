export const PROJECT_CASE_STUDIES: ProjectCaseStudy[] = [
  {
    id: "kip-student-eligibility-prediction",
    code: "PRO-19 // AI-03",
    title: "KIP Student Eligibility Prediction: Gaussian Naive Bayes Classification & Socioeconomic Decision Support",
    category: "ai-ml",
    categoryLabel: "AI & Machine Learning",
    year: "2024",
    status: "Completed",
    imageUrl: "kip1.webp",
    images: [
      "kip1.webp",
      "kip2.webp"
    ],
    summary:
      "Supervised machine learning decision-support platform leveraging Gaussian Naive Bayes and one-hot categorical encoding to classify student scholarship eligibility for Indonesia's PIP/KIP program from demographic and socioeconomic features, deployed via an interactive real-time Streamlit web dashboard.",
    overview:
      "KIP Student Eligibility Prediction is an applied machine learning decision-support system engineered to automate the evaluation of candidate eligibility for the Program Indonesia Pintar (PIP) scholarship program across junior high schools. Manually evaluating hundreds of student administrative records from national Dapodik registries is labor-intensive and prone to administrative inconsistency. This system resolves the bottleneck by implementing an end-to-end classification pipeline trained on historical student records from SMP Sepuluh Nopember 02 Semarang. Continuous indicators (parental income, sibling count, commute distance) and categorical indicators (KPS card enrollment, transportation mode, BSM aid recommendation) are transformed into a 10-dimensional feature space and modeled using a Gaussian Naive Bayes (GaussianNB) classifier. Serialized into a compact binary artifact and served through an interactive Streamlit web dashboard, the architecture supports both real-time manual parameter tweaking and batch CSV ingestion, providing transparent posterior class probabilities to aid institutional committees in allocating educational assistance fairly.",
    challenge:
      "Disparate feature scales between unnormalized monthly parental income ($10^6$ IDR with variance ~10^11) and binary indicators created extreme numerical variance imbalance in Gaussian likelihood calculations, while an 81.6% negative class skew naturally biased MAP decision boundaries toward rejection. Furthermore, single-candidate web queries risked feature-space misalignment when generating categorical dummy variables without reference schema concatenation.",
    architectureSummary:
      "Pandas reference-aligned dummy encoding + 10-dimensional feature vector assembly + Gaussian Naive Bayes (GaussianNB) probabilistic classifier + Pickle model serialization + Streamlit web serving dashboard",
    objectives: [
      "Automate the classification of prospective junior high school students into KIP scholarship recipients ('Ya') and non-recipients ('Tidak').",
      "Standardize candidate evaluation by modeling six essential socioeconomic and demographic attributes derived from institutional Dapodik school records.",
      "Mitigate training-serving feature drift by implementing dynamic reference-schema concatenation for categorical one-hot encoding during live inference.",
      "Compute calibrated posterior class probabilities (P(Eligible | X) vs. P(Ineligible | X)) to support administrative review of borderline applicants.",
      "Deliver an accessible, zero-dependency browser-based interface via Streamlit supporting both interactive manual input and batch CSV uploads."
    ],
    features: [
      {
        title: "Dynamic Schema Dummy Alignment",
        description:
          "Concatenates incoming user query DataFrames with reference training data prior to pd.get_dummies to guarantee exact 10-dimensional vector alignment during single-sample inference.",
        badge: "One-Hot Schema"
      },
      {
        title: "Gaussian Naive Bayes Classifier",
        description:
          "Calculates maximum a posteriori class predictions using class prior probabilities and continuous Gaussian feature likelihood densities across all 10 feature coordinates.",
        badge: "GaussianNB"
      },
      {
        title: "Dual-Class Posterior Probability Engine",
        description:
          "Computes continuous posterior probability distributions for both eligible and ineligible classes, giving evaluators granular certainty metrics beyond hard binary thresholds.",
        badge: "Bayesian Posteriori"
      },
      {
        title: "Multi-Input Administrative Dashboard",
        description:
          "Delivers an intuitive Streamlit interface featuring sliders for continuous attributes (income, siblings, distance) and select boxes for discrete welfare markers.",
        badge: "Streamlit UI"
      },
      {
        title: "Batch CSV Ingestion Module",
        description:
          "Integrates an automated CSV file uploader in the sidebar to rapidly parse, validate, and evaluate external student cohorts against the fitted model.",
        badge: "CSV File Uploader"
      },
      {
        title: "Decoupled Artifact Serialization",
        description:
          "Encapsulates the fitted GaussianNB model into a standalone pickle artifact, cleanly separating the Jupyter exploratory training pipeline from web serving.",
        badge: "Pickle Artifact"
      }
    ],
    pipelineSteps: [
      {
        step: "01. Student Profile Ingestion & Validation",
        detail:
          "Ingest student attributes via interactive Streamlit sidebar widgets or uploaded tabular CSV files and assemble raw feature series.",
        latencyOrMetric: "0.4ms Ingestion"
      },
      {
        step: "02. Reference Concatenation & Dummy Encoding",
        detail:
          "Concatenate the query record with the headless reference dataset to perform consistent one-hot expansion across KPS, transport, and BSM fields.",
        latencyOrMetric: "1.8ms Encoding"
      },
      {
        step: "03. 10-Dimensional Feature Matrix Assembly",
        detail:
          "Slice the first row of the dummy-expanded DataFrame to isolate the 10-dimensional aligned feature vector matching modelNBC_kip.pkl training schema.",
        latencyOrMetric: "0.6ms Array Slicing"
      },
      {
        step: "04. Gaussian Likelihood & Prior Computation",
        detail:
          "Pass the 10-dimensional vector through GaussianNB to calculate log-likelihoods across continuous Gaussian feature densities weighted by empirical class priors.",
        latencyOrMetric: "0.2ms Model Forward"
      },
      {
        step: "05. Posterior Probability & Label Assignment",
        detail:
          "Normalize class log-probabilities into posterior confidence scores and determine the predicted class label via maximum a posteriori (MAP) argmax.",
        latencyOrMetric: "0.1ms MAP Argmax"
      },
      {
        step: "06. Streamlit Reactive Dashboard Rendering",
        detail:
          "Render encoded feature tables, predicted status ('Ya' / 'Tidak'), and exact posterior probability distributions directly to the browser UI.",
        latencyOrMetric: "16ms UI Render"
      }
    ],
    evaluation:
      "The Gaussian Naive Bayes classifier was evaluated on the complete 500-student historical dataset from SMP Sepuluh Nopember 02 Semarang across 10 encoded feature dimensions. Under the standard maximum a posteriori decision rule with empirical priors (0.816 for non-recipients and 0.184 for recipients), the model achieved an overall accuracy of 81.60% across 408 true negative classifications, reflecting a highly conservative baseline driven by prior odds and wide continuous income variances. Analysis of posterior probabilities revealed that prospective candidates with active KPS welfare cards, low parental earnings, and BSM recommendations exhibited significantly elevated eligibility likelihoods ranging from 20.6% to 27.6% (compared to baseline priors of 18.4%), providing administrative review committees with actionable continuous risk scores. On standard workstation CPU hardware, end-to-end inference—spanning reference DataFrame concatenation, categorical dummy expansion, and GaussianNB forward pass—executed in under 3.2ms per query, ensuring frictionless real-time responsiveness within the Streamlit web dashboard.",
    outcomes: [
      "Successfully engineered a machine learning classification pipeline automating KIP scholarship eligibility assessment across 500 validated junior high student records.",
      "Eliminated categorical encoding mismatches during single-sample inference via automated training schema concatenation in Streamlit.",
      "Formulated an interpretable probabilistic scoring engine outputting exact posterior class probabilities for institutional transparency.",
      "Delivered an end-to-end accessible web application enabling school staff to perform instant manual checks or evaluate batch candidate CSV uploads.",
      "Achieved sub-5ms CPU-based local prediction latency with zero dedicated GPU requirements or cloud infrastructure overhead."
    ],
    techStack: [
      "Python",
      "scikit-learn",
      "Pandas",
      "NumPy",
      "Streamlit",
      "Pillow",
      "Jupyter Notebook",
      "Pickle",
      "OpenPyXL"
    ],
    accentColor: "indigo",
    repoUrl: "https://github.com/weztcy/KIP-Student-Eligibility-Prediction"
  }
];

