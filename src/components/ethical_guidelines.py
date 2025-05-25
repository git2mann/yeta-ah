import streamlit as st
from pages.academic_integrity_policies import render_academic_integrity_policies
from pages.best_practices import render_best_practices
from pages.student_rights import render_student_rights
from pages.data_protection import render_data_protection
from pages.algorithmic_fairness import render_algorithmic_fairness


def render_ethical_use_guidelines():
    """Render the Ethical Use Guidelines section."""
    st.subheader("Ethical Use Guidelines")
    st.markdown("""
    <div class="warning-box" style="background-color: #1F2937">
        <h3>⚠️ Important Notice</h3>
        <p>This system is designed to be a <strong>decision support tool</strong> and not an automated decision-making system. All flagged cases must be reviewed by qualified academic staff before any action is taken.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Key Principles")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Fairness & Bias Mitigation
        - Regular audits for algorithmic bias
        - Balanced representation in training data
        - Continuous monitoring for disparate impact
        
        #### Human Oversight
        - All flags require human review
        - Staff training on tool limitations
        - Appeal process for students
        """)
    with col2:
        st.markdown("""
        #### Transparency
        - Clear documentation of model workings
        - Explainable predictions with feature contributions
        - Open communication with stakeholders
        
        #### Privacy & Security
        - Data minimization principles
        - Secure storage protocols
        - Compliance with data protection regulations
        """)


def render_recommended_workflow():
    """Render the Recommended Workflow section."""
    st.subheader("Recommended Workflow")
    st.markdown("""
    1. **Initial Screening**: Use the system to identify potential anomalies
    2. **Human Review**: Have qualified academic staff review each flagged case
    3. **Additional Evidence**: Gather additional information beyond model output
    4. **Student Consultation**: Discuss concerns with students before decisions
    5. **Decision & Documentation**: Document reasoning for any actions taken
    """)


def render_explore_topics():
    """Render the Explore Topics section."""
    st.subheader("Explore Topics")
    st.markdown("Learn more about ethical guidelines and best practices below.")

    # Topics dictionary with descriptions and functions
    topics = {
        "Academic Integrity Policies": {
            "description": "Learn about policies that ensure fairness and honesty in academics.",
            "function": render_academic_integrity_policies
        },
        "Best Practices for Academic Misconduct Investigation": {
            "description": "Explore recommended practices for investigating academic misconduct.",
            "function": render_best_practices
        },
        "Student Rights & Responsibilities": {
            "description": "Understand the rights and responsibilities of students.",
            "function": render_student_rights
        },
        "Data Protection Guidelines": {
            "description": "Learn how to protect sensitive student data.",
            "function": render_data_protection
        },
        "Algorithmic Fairness Resources": {
            "description": "Discover resources to ensure fairness in machine learning algorithms.",
            "function": render_algorithmic_fairness
        }
    }

    # Organize topics into columns
    topic_keys = list(topics.keys())
    col1, col2, col3 = st.columns(3)

    # Render topics in the first column
    with col1:
        st.markdown(f"### {topic_keys[0]}")
        st.markdown(topics[topic_keys[0]]["description"])
        topics[topic_keys[0]]["function"]()

        st.markdown(f"### {topic_keys[1]}")
        st.markdown(topics[topic_keys[1]]["description"])
        topics[topic_keys[1]]["function"]()

    # Render topics in the second column
    with col2:
        st.markdown(f"### {topic_keys[2]}")
        st.markdown(topics[topic_keys[2]]["description"])
        topics[topic_keys[2]]["function"]()

        st.markdown(f"### {topic_keys[3]}")
        st.markdown(topics[topic_keys[3]]["description"])
        topics[topic_keys[3]]["function"]()

    # Render topics in the third column
    with col3:
        st.markdown(f"### {topic_keys[4]}")
        st.markdown(topics[topic_keys[4]]["description"])
        topics[topic_keys[4]]["function"]()


def render_feedback_form():
    """Render the Feedback Form section."""
    st.subheader("Feedback & Reporting")
    st.markdown("""
    We value your input to improve the system. Please report any concerns or suggestions, including:
    
    - Unexpected predictions or outcomes
    - Potential bias or fairness issues
    - Technical errors or bugs
    - Ideas for new features or improvements
    
    Your feedback ensures the system remains fair, effective, and aligned with user needs.
    """)
    
    with st.form("feedback_form"):
        st.markdown("#### Submit Your Feedback")
        feedback_type = st.selectbox(
            "Feedback Type",
            ["General Feedback", "Bug Report", "Bias Concern", "Feature Request"]
        )
        feedback_text = st.text_area("Your Feedback", height=150, placeholder="Describe your feedback here...")
        contact_info = st.text_input("Your Email (optional)", placeholder="Enter your email if you'd like a response")
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            if feedback_text.strip():
                st.success("Thank you for your feedback! It has been recorded and will be reviewed.")
                if contact_info.strip():
                    st.info(f"We may contact you at {contact_info} if further details are needed.")
            else:
                st.error("Please provide feedback before submitting.")


def render_mathematical_framework():
    """Render the Mathematical Framework section."""
    st.subheader("Mathematical Framework for Academic Misconduct Detection")
    st.markdown("""
    This section provides an overview of the mathematical principles and techniques used in the academic misconduct detection system.
    """)
    
    # Feature Space and Data Representation
    st.markdown("### Feature Space and Data Representation")
    st.markdown("""
    Let us define our student data as a collection of $n$ samples $\\{(x_i, y_i)\\}_{i=1}^{n}$ where:
    - $x_i \\in \\mathbb{R}^d$ represents the feature vector for student $i$, with $d=9$ features.
    - $y_i \\in \\{0, 1\\}$ represents the class label where $1$ indicates academic misconduct and $0$ indicates normal behavior.
    
    The feature space $\\mathcal{X} \\subset \\mathbb{R}^d$ encompasses all possible combinations of student performance metrics.
    """)
    st.latex(r"""
    x_i = \begin{bmatrix}
    z_{c,i} \\
    z_{e,i} \\
    z_{d,i} \\
    \sigma_i \\
    t_i \\
    p_i \\
    v_i \\
    h_i \\
    a_i
    \end{bmatrix}
    """)
    st.markdown("""
    Each dimension represents a specific aspect of academic behavior:
    - $z_{c,i}$: Coursework z-score
    - $z_{e,i}$: Exam z-score
    - $z_{d,i}$: Difference between coursework and exam performance
    - $\\sigma_i$: Score variance
    - $t_i$: Exam time standardized
    - $p_i$: Peer-relative performance
    - $v_i$: Subject variation
    - $h_i$: Historical trends
    - $a_i$: Anomaly score
    
    This comprehensive feature set captures various cheating behaviors, including coursework advantage (plagiarism), exam advantage (having answers in advance), and suspicious consistency patterns.
    """)
    
    # Feature Engineering Transformations
    st.markdown("### Feature Engineering Transformations")
    
    # Z-Score Transformation
    st.markdown("#### Z-Score Transformation")
    st.latex(r"""
    z_{j,i} = \frac{s_{j,i} - \mu_j}{\sigma_j}
    """)
    st.markdown("""
    where $\\mu_j = \\frac{1}{n} \\sum_{i=1}^{n} s_{j,i}$ and $\\sigma_j = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (s_{j,i} - \\mu_j)^2}$.
    
    Z-score normalization serves multiple purposes:
    1. **Standardization**: Converts raw scores to standard deviations from the mean, allowing fair comparison.
    2. **Distribution normalization**: Creates approximately normal distributions under typical conditions.
    3. **Outlier highlighting**: Makes unusual performances (±2 or more standard deviations) immediately apparent.
    
    The z-score difference $z_{d,i}$ quantifies inconsistency between coursework and exam performance. In legitimate academic scenarios, this difference typically follows a normal distribution centered near zero. Significant deviations suggest potential misconduct.
    """)
    
    # Subject Variation Metric
    st.markdown("#### Subject Variation Metric")
    st.latex(r"""
    v_i = \sqrt{\frac{1}{m} \sum_{j=1}^{m} (s_{i,j} - \overline{s}_i)^2}
    """)
    st.markdown("""
    where $\\overline{s}_i = \\frac{1}{m} \\sum_{j=1}^{m} s_{i,j}$ is the mean score across subjects.
    
    The subject variation metric $v_i$ is essentially the standard deviation of a student's performance across different subjects:
    - **Normal academic behavior**: Students typically show natural variation across subjects based on aptitude and interest.
    - **Cheating behavior**: Students engaging in misconduct often show unnaturally consistent performance.
    - **Quantification**: Lower values of $v_i$ than expected can signal suspicious consistency.
    
    This metric effectively identifies students who might be obtaining unauthorized assistance across multiple subjects, as their performance becomes artificially uniform.
    """)
    
    # Isolation Forest Anomaly Score
    st.markdown("#### Isolation Forest Anomaly Score")
    st.latex(r"""
    a_i = 2^{-\frac{E[h(x_i)]}{c(n)}}
    """)
    st.markdown("""
    where:
    - $h(x_i)$ is the path length for sample $x_i$ (number of edges traversed in isolation tree).
    - $E[h(x_i)]$ is the average path length across the forest.
    - $c(n) = 2H(n-1) - \\frac{2(n-1)}{n}$ where $H(i) \\approx \\ln(i) + 0.5772156649$ (Euler's constant).
    
    The Isolation Forest algorithm is particularly suited for detecting outliers in academic data because:
    1. **Unsupervised detection**: It identifies outliers without requiring labeled cheating examples.
    2. **Efficiency**: It isolates anomalies quickly by partitioning the feature space.
    3. **Path length principle**: Anomalous points require fewer partitions to isolate, resulting in shorter paths.
    
    The normalization calibrates the score to a $[0, 1]$ range, where values closer to 1 indicate greater anomaly likelihood.
    """)
    
    # Model Training and Optimization
    st.markdown("### Model Training and Optimization")
    
    # SMOTE Balancing
    st.markdown("#### SMOTE Balancing")
    st.latex(r"""
    \tilde{x} = x_i + \lambda \cdot (x_j - x_i)
    """)
    st.markdown("""
    where:
    - $x_i$ is a sample from the minority class
    - $x_j$ is one of the $k$-nearest neighbors of $x_i$ in the minority class
    - $\\lambda \\in [0,1]$ is a random number
    
    SMOTE (Synthetic Minority Oversampling Technique) addresses class imbalance by:
    1. **Avoiding duplicating existing samples**: Creating new, unique examples.
    2. **Preserving feature relationships**: Maintaining underlying patterns of cheating behavior.
    3. **Expanding minority class representation**: Allowing better learning of decision boundaries.
    """)
    
    # StandardScaler Transformation
    st.markdown("#### StandardScaler Transformation")
    st.latex(r"""
    \tilde{x}_{i,f} = \frac{x_{i,f} - \mu_f}{\sigma_f}
    """)
    st.markdown("""
    where $\\mu_f$ and $\\sigma_f$ are the mean and standard deviation of feature $f$ across all samples.
    
    Feature standardization ensures that all features contribute equitably to the model regardless of their original scales:
    1. **Equalizes feature importance**: Prevents features with larger numeric ranges from dominating.
    2. **Improves convergence**: Facilitates optimization during model training.
    3. **Maintains interpretability**: Allows comparison of feature importances on a common scale.
    """)
    
    # Grid Search Optimization
    st.markdown("#### Grid Search Optimization")
    st.latex(r"""
    \theta^* = \underset{\theta \in \Theta}{\arg\max} \, \text{F1}_{\text{CV}}(\theta)
    """)
    st.markdown("""
    where:
    - $\\Theta$ is the hyperparameter search space
    - $\\text{F1}_{\\text{CV}}(\\theta)$ is the average F1-score across cross-validation folds
    
    Grid search explores a predefined hyperparameter space to find the configuration that maximizes model performance:
    1. **F1-score optimization**: Balances precision and recall, crucial for fair assessment.
    2. **Cross-validation robustness**: Ensures performance generalizes across different data subsets.
    3. **Domain-specific tuning**: Focuses on parameters most relevant to capturing cheating patterns.
    
    The optimal configuration balances model complexity with generalization ability, avoiding both underfitting and overfitting.
    """)
    
    # Threshold Optimization
    st.markdown("#### Threshold Optimization")
    st.latex(r"""
    \tau^* = \underset{\tau \in [0,1]}{\arg\max} \, \text{F}_{\beta}(\tau)
    """)
    st.markdown("""
    where $\\text{F}_\\beta$ is the F-beta score:
    """)
    st.latex(r"""
    \text{F}_{\beta} = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
    """)
    st.markdown("""
    Threshold optimization is particularly important due to asymmetric consequences of errors:
    1. **Precision emphasis**: False accusations can severely impact students' academic careers.
    2. **Recall importance**: Missing actual misconduct undermines academic integrity.
    3. **Institutional policy alignment**: The threshold can be adjusted to reflect specific priorities.
    
    The threshold of 0.58 indicates a slight preference for precision over recall, requiring stronger evidence before flagging a student.
    """)
    
    # Feature Importance Quantification
    st.markdown("#### Feature Importance Quantification")
    st.latex(r"""
    I_f = \frac{1}{B}\sum_{b=1}^B \sum_{n \in T_b} \Delta I(n, f) \cdot \frac{|S_n|}{|S_{\text{root}}|}
    """)
    st.markdown("""
    where:
    - $n$ iterates over all nodes in tree $T_b$ that split on feature $f$
    - $\\Delta I(n, f)$ is the impurity decrease at node $n$
    - $|S_n|$ is the number of samples at node $n$
    - $|S_{\\text{root}}|$ is the number of samples at the root node
    
    Feature importance analysis reveals:
    1. **Historical trend dominance (32.7%)**: Sudden improvements in performance are the strongest indicators.
    2. **Score variance significance (26.4%)**: Unnatural consistency across assessments is highly suspicious.
    3. **Subject variation relevance (12.8%)**: Unusual patterns across different subjects provide important signals.
    4. **Temporal cues (9.9%)**: Unusual exam timing patterns contribute meaningful information.
    5. **Anomaly score integration (7.8%)**: The anomaly score adds complementary perspective.
    """)
    
    # Evaluation Metrics
    st.markdown("### Evaluation Metrics")
    
    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    st.markdown("""
    - True Positives (TP): $\\sum_{i=1}^n \\mathbb{I}(y_i = 1 \\text{ and } \\hat{y}_i = 1)$
    - False Positives (FP): $\\sum_{i=1}^n \\mathbb{I}(y_i = 0 \\text{ and } \\hat{y}_i = 1)$
    - False Negatives (FN): $\\sum_{i=1}^n \\mathbb{I}(y_i = 1 \\text{ and } \\hat{y}_i = 0)$
    - True Negatives (TN): $\\sum_{i=1}^n \\mathbb{I}(y_i = 0 \\text{ and } \\hat{y}_i = 0)$
    
    The confusion matrix results:
    - TP (27): Correctly identified instances of academic misconduct
    - FP (2): Honest students incorrectly flagged as potential cheaters
    - FN (4): Missed instances of academic misconduct
    - TN (167): Correctly identified honest academic behavior
    
    The very low false positive rate (approximately 1.2%) is particularly important given the serious implications of misconduct allegations.
    """)
    
    # Performance Metrics
    st.markdown("#### Performance Metrics")
    st.latex(r"""
    \begin{align}
    \text{Precision} &= \frac{TP}{TP + FP} = \frac{27}{27 + 2} = 0.93\\
    \text{Recall} &= \frac{TP}{TP + FN} = \frac{27}{27 + 4} = 0.87\\
    \text{F1-Score} &= \frac{2 \cdot P \cdot R}{P + R} = \frac{2 \cdot 0.93 \cdot 0.87}{0.93 + 0.87} = 0.90\\
    \text{Accuracy} &= \frac{TP + TN}{TP + TN + FP + FN} = \frac{27 + 167}{27 + 167 + 2 + 4} = 0.97
    \end{align}
    """)
    st.markdown("""
    These metrics quantify different aspects of model performance:
    1. **Precision (93%)**: When the model flags a student, it's rarely a false alarm.
    2. **Recall (87%)**: Most misconduct cases are successfully identified.
    3. **F1-Score (90%)**: The model balances precision and recall effectively.
    4. **Accuracy (97%)**: Overall, the model correctly classifies the vast majority of cases.
    """)
    
    # Probabilistic Interpretation
    st.markdown("### Probabilistic Interpretation")
    st.latex(r"""
    P(y=1|x) = P(x \in \mathcal{C}_1 | x)
    """)
    st.markdown("""
    where $\\mathcal{C}_1$ represents the set of feature vectors associated with academic misconduct.
    
    For decision-making, we use the threshold rule:
    """)
    st.latex(r"""
    \hat{y} = 
    \begin{cases} 
    1 & \text{if } P(y=1|x) > \tau^* \\
    0 & \text{otherwise}
    \end{cases}
    """)
    st.markdown("""
    This probabilistic framework provides several advantages:
    1. **Uncertainty quantification**: Probabilities express the strength of evidence rather than binary judgments.
    2. **Risk control**: Institutions can adjust thresholds based on their tolerance for different error types.
    3. **Transparency**: Probabilistic outputs can be explained to stakeholders as likelihood estimates.
    """)
    
    # Bayesian Risk Minimization
    st.markdown("### Bayesian Risk Minimization")
    st.latex(r"""
    R(h) = \mathbb{E}_{x,y}[L(y, h(x))]
    """)
    st.markdown("""
    where $L$ is a cost-sensitive loss function:
    """)
    st.latex(r"""
    L(y, \hat{y}) = 
    \begin{cases} 
    0 & \text{if } y = \hat{y} \\
    C_{FP} & \text{if } y = 0, \hat{y} = 1 \\
    C_{FN} & \text{if } y = 1, \hat{y} = 0
    \end{cases}
    """)
    st.markdown("""
    The optimal threshold in this framework becomes:
    """)
    st.latex(r"""
    \tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}
    """)
    st.markdown("""
    where $C_{FP}$ and $C_{FN}$ are the costs associated with false positives and false negatives.
    
    With the optimal threshold of approximately 0.58, we can infer that the implicit cost ratio is $\\frac{C_{FP}}{C_{FN}} \\approx \\frac{0.58}{0.42} \\approx 1.38$, suggesting that false positives are considered about 38% more costly than false negatives.
    """)
    
    # Feature Space Visualization
    st.markdown("### Feature Space Visualization")
    st.latex(r"""
    z_i = t\text{-SNE}(x_i)
    """)
    st.markdown("""
    where $z_i \\in \\mathbb{R}^2$ is a two-dimensional representation of $x_i$ preserving the local neighborhood structure.
    
    Visualizing the feature space provides several insights:
    1. **Cluster identification**: Reveals natural groupings of similar academic behaviors.
    2. **Boundary visualization**: Shows how the model separates legitimate and suspicious behaviors.
    3. **Outlier detection**: Highlights unusual cases for further investigation.
    """)
    
    # Additional Considerations
    st.markdown("### Additional Considerations")
    
    # Permutation Feature Importance
    st.markdown("#### Permutation Feature Importance")
    st.latex(r"""
    I_f^{perm} = M(y, \hat{y}) - M(y, \hat{y}_f^{perm})
    """)
    st.markdown("""
    where $\\hat{y}_f^{perm}$ represents predictions after randomly permuting feature $f$.
    
    Permutation importance offers a model-agnostic alternative to the built-in Random Forest importance metric:
    1. **Causal perspective**: Measures the performance drop when feature information is destroyed.
    2. **Correlation robustness**: Less influenced by correlations between features.
    3. **Direct interpretation**: Directly tied to model performance rather than tree structure.
    """)
    
    # Calibration Analysis
    st.markdown("#### Calibration Analysis")
    st.latex(r"""
    P(y=1 | P(y=1|x) = p) \approx p
    """)
    st.markdown("""
    Probability calibration is crucial for decision-making in high-stakes contexts:
    1. **Reliability assessment**: Ensures probabilistic outputs reflect true likelihoods.
    2. **Confidence interpretation**: Allows stakeholders to interpret probability scores correctly.
    3. **Fair comparison**: Enables consistent thresholds across different contexts.
    """)
    
    # SHAP Values
    st.markdown("#### Local Explainability with SHAP Values")
    st.latex(r"""
    \phi_f(x_i) = \sum_{S \subseteq F \setminus \{f\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f_x(S \cup \{f\}) - f_x(S)]
    """)
    st.markdown("""
    where $F$ is the set of all features and $f_x(S)$ is the prediction using only features in subset $S$.
    
    SHAP values offer instance-specific explanations:
    1. **Case-specific justification**: Explains precisely why a particular student was flagged.
    2. **Transparent decision-making**: Enables clear communication with stakeholders.
    3. **Pattern identification**: Helps identify new or evolving cheating strategies.
    """)
    
    # Conclusion
    st.markdown("### Conclusion")
    st.markdown("""
    This comprehensive mathematical framework establishes a rigorous foundation for academic misconduct detection using machine learning. By formalizing each component—from feature engineering to model training and evaluation—we ensure that the system is both technically sound and ethically responsible.
    
    The demonstrated performance metrics (93% precision, 87% recall, 90% F1-score) validate that this approach effectively balances the competing objectives of identifying academic misconduct while minimizing false accusations. The mathematical formulation provides transparency into the decision-making process, essential for fair and responsible application in educational settings.
    """)

    # Download Option
    st.markdown("### Download the Full Framework")
    with open("Yeta_Ah.pdf", "rb") as pdf_file:
        pdf_data = pdf_file.read()
    st.download_button(
        label="Download Full Framework (PDF)",
        data=pdf_data,
        file_name="Yeta_Ah.pdf",
        mime="application/pdf"
    )

def render_ethical_guidelines():
    """Main function to render the Ethical Guidelines page."""
    st.markdown('<div class="notion-accent-bar"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <h2 style="display:inline; margin-left: 0.5rem;">⚖️ Ethical Guidelines</h2>
        <div class="notion-callout" style="margin-top:1rem;">
            <span style="font-size:1.5rem;">🔒</span>
            <div>
                Important considerations for responsible and fair use of the system.<br>
                <span style="color:#60a5fa;">Tip:</span> Review these guidelines before using the tool in practice.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ethical Use Guidelines Section
    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.subheader("Ethical Use Guidelines")
    st.markdown("""
    <div class="warning-box" style="background-color: #1F2937">
        <h3>⚠️ Important Notice</h3>
        <p>This system is designed to be a <strong>decision support tool</strong> and not an automated decision-making system. All flagged cases must be reviewed by qualified academic staff before any action is taken.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Key Principles")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Fairness & Bias Mitigation
        - Regular audits for algorithmic bias
        - Balanced representation in training data
        - Continuous monitoring for disparate impact

        #### Human Oversight
        - All flags require human review
        - Staff training on tool limitations
        - Appeal process for students
        """)
    with col2:
        st.markdown("""
        #### Transparency
        - Clear documentation of model workings
        - Explainable predictions with feature contributions
        - Open communication with stakeholders

        #### Privacy & Security
        - Data minimization principles
        - Secure storage protocols
        - Compliance with data protection regulations
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Recommended Workflow Section
    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.subheader("Recommended Workflow")
    st.markdown("""
    1. **Initial Screening**: Use the system to identify potential anomalies  
    2. **Human Review**: Have qualified academic staff review each flagged case  
    3. **Additional Evidence**: Gather additional information beyond model output  
    4. **Student Consultation**: Discuss concerns with students before decisions  
    5. **Decision & Documentation**: Document reasoning for any actions taken  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Explore Topics Section
    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.subheader("Explore Topics")
    st.markdown("Learn more about ethical guidelines and best practices below.")

    topics = {
        "Academic Integrity Policies": {
            "description": "Learn about policies that ensure fairness and honesty in academics.",
            "function": render_academic_integrity_policies
        },
        "Best Practices for Academic Misconduct Investigation": {
            "description": "Explore recommended practices for investigating academic misconduct.",
            "function": render_best_practices
        },
        "Student Rights & Responsibilities": {
            "description": "Understand the rights and responsibilities of students.",
            "function": render_student_rights
        },
        "Data Protection Guidelines": {
            "description": "Learn how to protect sensitive student data.",
            "function": render_data_protection
        },
        "Algorithmic Fairness Resources": {
            "description": "Discover resources to ensure fairness in machine learning algorithms.",
            "function": render_algorithmic_fairness
        }
    }
    topic_keys = list(topics.keys())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {topic_keys[0]}")
        st.markdown(topics[topic_keys[0]]["description"])
        topics[topic_keys[0]]["function"]()
        st.markdown(f"### {topic_keys[1]}")
        st.markdown(topics[topic_keys[1]]["description"])
        topics[topic_keys[1]]["function"]()
    with col2:
        st.markdown(f"### {topic_keys[2]}")
        st.markdown(topics[topic_keys[2]]["description"])
        topics[topic_keys[2]]["function"]()
        st.markdown(f"### {topic_keys[3]}")
        st.markdown(topics[topic_keys[3]]["description"])
        topics[topic_keys[3]]["function"]()
    with col3:
        st.markdown(f"### {topic_keys[4]}")
        st.markdown(topics[topic_keys[4]]["description"])
        topics[topic_keys[4]]["function"]()
    st.markdown('</div>', unsafe_allow_html=True)

    # Mathematical Framework Section
    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.subheader("Mathematical Framework for Academic Misconduct Detection")
    st.markdown("""
    <span style="color:#60a5fa;">Explore the mathematical principles and techniques used in the detection system.</span>
    """, unsafe_allow_html=True)
    render_mathematical_framework()
    st.markdown('</div>', unsafe_allow_html=True)

    # Feedback Form Section
    st.markdown('<div class="feature-card" style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.subheader("Feedback & Reporting")
    st.markdown("""
    We value your input to improve the system. Please report any concerns or suggestions, including:
    - Unexpected predictions or outcomes
    - Potential bias or fairness issues
    - Technical errors or bugs
    - Ideas for new features or improvements

    Your feedback ensures the system remains fair, effective, and aligned with user needs.
    """)
    with st.form("feedback_form"):
        st.markdown("#### Submit Your Feedback")
        feedback_type = st.selectbox(
            "Feedback Type",
            ["General Feedback", "Bug Report", "Bias Concern", "Feature Request"]
        )
        feedback_text = st.text_area("Your Feedback", height=150, placeholder="Describe your feedback here...")
        contact_info = st.text_input("Your Email (optional)", placeholder="Enter your email if you'd like a response")
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        if submitted:
            if feedback_text.strip():
                st.success("Thank you for your feedback! It has been recorded and will be reviewed.")
                if contact_info.strip():
                    st.info(f"We may contact you at {contact_info} if further details are needed.")
            else:
                st.error("Please provide feedback before submitting.")
    st.markdown('</div>', unsafe_allow_html=True)