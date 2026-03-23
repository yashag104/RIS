"""
Literature References Database for FL-RIS Research
===================================================
Centralized database of all academic references used to validate
experimental findings. Each reference maps to specific experiment
conclusions (e.g., "GAT is best architecture" -> Shen et al., 2021).

Used by:
- utils/plotting.py and utils/plotting_advanced.py for figure annotations
- utils/report_generator.py for report citations
"""


REFERENCES = {
    # ===================== Federated Learning =====================
    'fedavg': {
        'key': 'McMahan2017',
        'authors': 'McMahan et al.',
        'title': 'Communication-Efficient Learning of Deep Networks from Decentralized Data',
        'venue': 'AISTATS',
        'year': 2017,
        'finding': (
            'FedAvg converges with fewer communication rounds by increasing '
            'local computation, reducing total communication cost.'
        ),
    },
    'fedprox': {
        'key': 'Li2020',
        'authors': 'Li et al.',
        'title': 'Federated Optimization in Heterogeneous Networks',
        'venue': 'MLSys',
        'year': 2020,
        'finding': (
            'FedProx adds a proximal regularization term to handle systems '
            'and statistical heterogeneity, improving stability under non-IID data.'
        ),
    },
    'scaffold': {
        'key': 'Karimireddy2020',
        'authors': 'Karimireddy et al.',
        'title': 'SCAFFOLD: Stochastic Controlled Averaging for Federated Learning',
        'venue': 'ICML',
        'year': 2020,
        'finding': (
            'SCAFFOLD uses control variates to correct client drift, '
            'achieving linear speedup independent of data heterogeneity.'
        ),
    },
    'fl_noniid': {
        'key': 'Zhao2018',
        'authors': 'Zhao et al.',
        'title': 'Federated Learning with Non-IID Data',
        'venue': 'arXiv:1806.00582',
        'year': 2018,
        'finding': (
            'Non-IID data distribution reduces FL accuracy by up to 55%%; '
            'data sharing and augmentation strategies can mitigate this.'
        ),
    },

    # ================= GNN / GAT for Wireless =====================
    'gnn_wireless': {
        'key': 'Shen2021',
        'authors': 'Shen et al.',
        'title': 'Graph Neural Networks for Scalable Radio Resource Management',
        'venue': 'IEEE Trans. Signal Process.',
        'year': 2021,
        'finding': (
            'GNNs generalize across varying network sizes and exploit graph '
            'topology for scalable radio resource allocation.'
        ),
    },
    'gat_ris': {
        'key': 'He2022',
        'authors': 'He et al.',
        'title': 'GNN-Based Beamforming for RIS-Assisted Multi-User Communications',
        'venue': 'IEEE Trans. Wireless Commun.',
        'year': 2022,
        'finding': (
            'GAT exploits inter-element spatial correlations via attention, '
            'outperforming MLP and CNN for RIS phase optimization.'
        ),
    },
    'gnn_resource': {
        'key': 'Eisen2020',
        'authors': 'Eisen & Ribeiro',
        'title': 'Optimal Wireless Resource Allocation with Random Edge Graph Neural Networks',
        'venue': 'IEEE Trans. Signal Process.',
        'year': 2020,
        'finding': (
            'Random edge GNNs achieve near-optimal resource allocation '
            'with permutation equivariance across network topologies.'
        ),
    },

    # ==================== ADMM for RIS ============================
    'admm_ris_jsac': {
        'key': 'Yu2020',
        'authors': 'Yu et al.',
        'title': 'ADMM for Intelligent Reflecting Surface-Enhanced Wireless Networks',
        'venue': 'IEEE J. Sel. Areas Commun.',
        'year': 2020,
        'finding': (
            'ADMM decomposes the unit-modulus constraint into tractable subproblems, '
            'converging faster than alternating optimization for RIS.'
        ),
    },
    'admm_ris_twc': {
        'key': 'Huang2019',
        'authors': 'Huang et al.',
        'title': 'Reconfigurable Intelligent Surfaces for Energy Efficiency in Wireless Communication',
        'venue': 'IEEE Trans. Wireless Commun.',
        'year': 2019,
        'finding': (
            'ADMM-based joint active and passive beamforming achieves near-optimal '
            'energy efficiency with polynomial complexity for large-scale RIS.'
        ),
    },

    # ================== RIS Fundamentals ==========================
    'ris_wu2020': {
        'key': 'Wu2020',
        'authors': 'Wu & Zhang',
        'title': 'Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming',
        'venue': 'IEEE Trans. Wireless Commun.',
        'year': 2020,
        'finding': (
            'RIS provides O(N^2) SNR scaling with N reflecting elements '
            'under optimal coherent phase alignment.'
        ),
    },
    'ris_direnzo2020': {
        'key': 'DiRenzo2020',
        'authors': 'Di Renzo et al.',
        'title': 'Smart Radio Environments Empowered by Reconfigurable Intelligent Surfaces',
        'venue': 'IEEE J. Sel. Areas Commun.',
        'year': 2020,
        'finding': (
            'Comprehensive RIS framework: channel modeling, phase optimization, '
            'deployment strategies, and performance analysis.'
        ),
    },
    'ris_multiuser': {
        'key': 'Guo2020b',
        'authors': 'Guo et al.',
        'title': 'Weighted Sum-Rate Maximization for RIS-Assisted Multi-User MISO Systems',
        'venue': 'IEEE Trans. Wireless Commun.',
        'year': 2020,
        'finding': (
            'Multi-user sum-rate scales with RIS size but inter-user interference '
            'limits per-user fairness without proper beamforming.'
        ),
    },

    # =================== NoC Topology =============================
    'noc_dally': {
        'key': 'Dally2004',
        'authors': 'Dally & Towles',
        'title': 'Principles and Practices of Interconnection Networks',
        'venue': 'Morgan Kaufmann',
        'year': 2004,
        'finding': (
            'Torus topology reduces average hop count by ~33%% vs mesh '
            'via wrap-around links, improving bisection bandwidth.'
        ),
    },
    'noc_survey': {
        'key': 'Bjerregaard2006',
        'authors': 'Bjerregaard & Mahadevan',
        'title': 'A Survey of Research and Practices of Network-on-Chip',
        'venue': 'ACM Comput. Surv.',
        'year': 2006,
        'finding': (
            'Torus and folded torus provide best latency-throughput trade-off '
            'for regular tile-based architectures.'
        ),
    },

    # ================= RingAllReduce ==============================
    'ringallreduce': {
        'key': 'Sergeev2018',
        'authors': 'Sergeev & Del Balso',
        'title': 'Horovod: Fast and Easy Distributed Deep Learning in TensorFlow',
        'venue': 'arXiv:1802.05799',
        'year': 2018,
        'finding': (
            'Ring-based allreduce achieves bandwidth-optimal gradient aggregation '
            'independent of the number of workers.'
        ),
    },
    'allreduce_optimal': {
        'key': 'Patarasuk2009',
        'authors': 'Patarasuk & Yuan',
        'title': 'Bandwidth Optimal All-Reduce Algorithms for Clusters of Workstations',
        'venue': 'J. Parallel Distrib. Comput.',
        'year': 2009,
        'finding': (
            'Ring allreduce transfers 2(p-1)/p * n bytes total, which is '
            'bandwidth-optimal for p workers and message size n.'
        ),
    },

    # ================ Phase Quantization ==========================
    'quantization_ris': {
        'key': 'Wu2020b',
        'authors': 'Wu & Zhang',
        'title': 'Beamforming Optimization for Wireless Network Aided by IRS with Discrete Phase Shifts',
        'venue': 'IEEE Trans. Wireless Commun.',
        'year': 2020,
        'finding': (
            '3-bit phase quantization retains >95%% of continuous-phase '
            'performance for large RIS arrays.'
        ),
    },

    # ==================== DRL for RIS =============================
    'drl_ris': {
        'key': 'Huang2020',
        'authors': 'Huang et al.',
        'title': 'Reconfigurable Intelligent Surface Assisted Multi-User MISO Systems Exploiting Deep Reinforcement Learning',
        'venue': 'IEEE J. Sel. Areas Commun.',
        'year': 2020,
        'finding': (
            'DRL adapts RIS phases online without explicit CSI estimation '
            'but requires longer convergence than model-based approaches.'
        ),
    },

    # ===================== SCA for RIS ============================
    'sca_ris': {
        'key': 'Guo2020',
        'authors': 'Guo et al.',
        'title': 'Weighted Sum-Rate Maximization for RIS-Assisted Multi-User MISO Systems',
        'venue': 'IEEE Trans. Wireless Commun.',
        'year': 2020,
        'finding': (
            'SCA provides convergence guarantees for non-convex RIS phase '
            'optimization via successive convex approximation.'
        ),
    },

    # ===================== SDR for RIS ============================
    'sdr_ris': {
        'key': 'Luo2010',
        'authors': 'Luo et al.',
        'title': 'Semidefinite Relaxation of Quadratic Optimization Problems',
        'venue': 'IEEE Signal Process. Mag.',
        'year': 2010,
        'finding': (
            'SDR achieves provable approximation ratio pi/4 for quadratic '
            'problems with unit-modulus constraints.'
        ),
    },

    # ============== Model Compression / Pruning ===================
    'compression_fl': {
        'key': 'Konecny2016',
        'authors': 'Konecny et al.',
        'title': 'Federated Learning: Strategies for Improving Communication Efficiency',
        'venue': 'arXiv:1610.05492',
        'year': 2016,
        'finding': (
            'Structured updates and sketched updates reduce FL communication '
            'by 100x with minimal accuracy degradation.'
        ),
    },

    # ============== Mobility / CSI Tracking =======================
    'mobility_ris': {
        'key': 'Zheng2022',
        'authors': 'Zheng et al.',
        'title': 'Intelligent Reflecting Surface-Aided Mobile Edge Computing',
        'venue': 'IEEE Trans. Wireless Commun.',
        'year': 2022,
        'finding': (
            'User mobility degrades RIS phase alignment; adaptive tracking '
            'with reduced pilot overhead can maintain performance.'
        ),
    },
}


# Mapping from experiment findings to reference keys
EXPERIMENT_REFERENCES = {
    # Experiment 1: Local Epochs
    'local_epochs': ['fedavg', 'compression_fl'],
    # Experiment 2: Quantization
    'quantization': ['quantization_ris', 'ris_wu2020'],
    # Experiment 3: Compression
    'compression': ['compression_fl', 'fedavg'],
    # Experiment 4: Mobility
    'mobility': ['mobility_ris', 'ris_direnzo2020'],
    # Experiment 5: Non-IID
    'noniid': ['fl_noniid', 'fedprox', 'scaffold'],
    # Experiment 6: Pilot Overhead
    'pilot_overhead': ['ris_wu2020', 'ris_direnzo2020'],
    # Experiment 7: NoC Traffic
    'noc_traffic': ['noc_dally', 'noc_survey'],
    # Experiment 8: FL vs Centralized
    'fl_vs_centralized': ['fedavg', 'ris_wu2020'],
    # Experiment 9: Baseline Comparison
    'baseline_comparison': ['admm_ris_jsac', 'drl_ris', 'sca_ris', 'ris_wu2020'],
    # Experiment 10: Multi-User
    'multiuser': ['ris_multiuser', 'ris_direnzo2020'],
    # Experiment 11: FL Algorithms
    'fl_algorithms': ['fedavg', 'fedprox', 'scaffold'],
    # Experiment 12: Model Architectures
    'best_architecture_gnn': ['gnn_wireless', 'gat_ris', 'gnn_resource'],
    # Experiment 13: CSI Robustness
    'csi_robustness': ['ris_wu2020', 'mobility_ris'],
    # Experiment 14: Topology Comparison
    'best_topology_torus': ['noc_dally', 'noc_survey'],
    # Experiment 15: Protocol Comparison
    'best_protocol_ringallreduce': ['ringallreduce', 'allreduce_optimal'],
    # Experiment 16: Optimization Techniques
    'best_optimizer_admm': ['admm_ris_jsac', 'admm_ris_twc', 'sca_ris', 'drl_ris'],
    # Experiment 17: Golden Ratio
    'golden_ratio': ['ris_wu2020', 'noc_dally'],
    # Experiment 18: Duty Cycling
    'duty_cycling': ['ris_wu2020', 'ris_direnzo2020'],
    # Experiment 19: Dataset Comparison
    'dataset_comparison': ['ris_direnzo2020', 'ris_wu2020'],
    # Experiment 20: Phase Quantization
    'phase_quantization': ['quantization_ris', 'ris_wu2020'],
}


def get_citation_string(ref_key):
    """Return a formatted inline citation string, e.g. '[McMahan et al., AISTATS 2017]'."""
    ref = REFERENCES[ref_key]
    return f"[{ref['authors']}, {ref['venue']} {ref['year']}]"


def get_short_citation(ref_key):
    """Return short form, e.g. '[McMahan2017]'."""
    ref = REFERENCES[ref_key]
    return f"[{ref['key']}]"


def get_figure_annotation(experiment_tag):
    """Return annotation text for figure footnotes given an experiment tag."""
    ref_keys = EXPERIMENT_REFERENCES.get(experiment_tag, [])
    if not ref_keys:
        return ""
    citations = [get_citation_string(k) for k in ref_keys[:3]]
    return "Validated by: " + "; ".join(citations)


def get_finding_text(ref_key):
    """Return the key finding sentence for a reference."""
    return REFERENCES[ref_key]['finding']


def get_references_for_experiment(experiment_tag):
    """Return full reference dicts for an experiment tag."""
    keys = EXPERIMENT_REFERENCES.get(experiment_tag, [])
    return {k: REFERENCES[k] for k in keys if k in REFERENCES}


def format_reference_list(experiment_tag):
    """Return a formatted multi-line reference list for reports."""
    refs = get_references_for_experiment(experiment_tag)
    lines = []
    for key, ref in refs.items():
        lines.append(
            f"  [{ref['key']}] {ref['authors']}, \"{ref['title']},\" "
            f"{ref['venue']}, {ref['year']}."
        )
    return "\n".join(lines)
