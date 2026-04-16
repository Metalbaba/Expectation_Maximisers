import torch

class DawidSkeneEM:
    def __init__(self, num_prompts: int, num_annotators: int, device: str = "cpu"):
        """
        Initializes the Dawid-Skene EM algorithm for binary preferences.
        """
        self.num_prompts = num_prompts
        self.num_annotators = num_annotators
        self.device = torch.device(device)
        
        # Initialize priors: assume 50/50 chance A or B is better globally
        self.pi_1 = torch.tensor(0.5, device=self.device)
        
        # Initialize annotator parameters: Assume humans are generally slightly better than random
        # alpha: P(vote A | truth is A), beta: P(vote B | truth is B)
        self.alpha = torch.full((num_annotators,), 0.7, dtype=torch.float32, device=self.device)
        self.beta = torch.full((num_annotators,), 0.7, dtype=torch.float32, device=self.device)
        
        # gamma[i] will store our calculated probability that Z_i = 1 (A is better)
        self.gamma = torch.full((num_prompts,), 0.5, dtype=torch.float32, device=self.device)

    def e_step(self, prompts: torch.Tensor, annotators: torch.Tensor, votes: torch.Tensor, llm_priors: torch.Tensor = None):
        """
        Calculates the posterior probabilities of the true labels (gamma).
        prompts: 1D tensor of prompt IDs
        annotators: 1D tensor of annotator IDs
        votes: 1D tensor of votes (1 for A, 0 for B)
        """
        # Calculate log-likelihoods to prevent numerical underflow with large products
        log_prob_A_is_true = torch.zeros(self.num_prompts, device=self.device)
        log_prob_B_is_true = torch.zeros(self.num_prompts, device=self.device)

        # DYNAMIC PRIOR INJECTION:
        if llm_priors is not None:
            # llm_priors is a tensor of shape (num_prompts,) with values between 0 and 1
            prior_A = llm_priors
            prior_B = 1.0 - llm_priors
        else:
            # Fallback to the original static 50/50 prior
            prior_A = self.pi_1
            prior_B = 1.0 - self.pi_1

        # Add the priors
        log_prob_A_is_true += torch.log(prior_A + 1e-9)
        log_prob_B_is_true += torch.log(prior_B + 1e-9)
        
        # For votes = 1 (Annotator chose A)
        mask_1 = (votes == 1)
        if mask_1.any():
            p_ids_1 = prompts[mask_1]
            a_ids_1 = annotators[mask_1]
            # If truth is A and vote is 1 -> probability is alpha
            log_prob_A_is_true.scatter_add_(0, p_ids_1, torch.log(self.alpha[a_ids_1]))
            # If truth is B and vote is 1 -> probability is (1 - beta)
            log_prob_B_is_true.scatter_add_(0, p_ids_1, torch.log(1.0 - self.beta[a_ids_1]))

        # For votes = 0 (Annotator chose B)
        mask_0 = (votes == 0)
        if mask_0.any():
            p_ids_0 = prompts[mask_0]
            a_ids_0 = annotators[mask_0]
            # If truth is A and vote is 0 -> probability is (1 - alpha)
            log_prob_A_is_true.scatter_add_(0, p_ids_0, torch.log(1.0 - self.alpha[a_ids_0]))
            # If truth is B and vote is 0 -> probability is beta
            log_prob_B_is_true.scatter_add_(0, p_ids_0, torch.log(self.beta[a_ids_0]))

        # Add global prior
        log_prob_A_is_true += torch.log(self.pi_1)
        log_prob_B_is_true += torch.log(1.0 - self.pi_1)

        # Convert back from log-space safely
        max_log = torch.maximum(log_prob_A_is_true, log_prob_B_is_true)
        prob_A = torch.exp(log_prob_A_is_true - max_log)
        prob_B = torch.exp(log_prob_B_is_true - max_log)

        # Update gamma: Normalized probability that A is the true winner
        self.gamma = prob_A / (prob_A + prob_B)

    def m_step(self, prompts: torch.Tensor, annotators: torch.Tensor, votes: torch.Tensor):
        """
        Updates the annotator accuracy parameters (alpha, beta) and global prior (pi_1).
        """
        # 1. Update global prior
        self.pi_1 = self.gamma.mean()

        # 2. Update alpha and beta for each annotator
        alpha_num = torch.zeros(self.num_annotators, device=self.device)
        alpha_den = torch.zeros(self.num_annotators, device=self.device)
        beta_num = torch.zeros(self.num_annotators, device=self.device)
        beta_den = torch.zeros(self.num_annotators, device=self.device)

        # Vectorized accumulation using scatter_add
        gamma_vals = self.gamma[prompts]
        
        # Denominators: Total expected times truth was A (for alpha) or B (for beta)
        alpha_den.scatter_add_(0, annotators, gamma_vals)
        beta_den.scatter_add_(0, annotators, 1.0 - gamma_vals)

        # Numerators: Expected times they voted correctly
        mask_1 = (votes == 1)
        alpha_num.scatter_add_(0, annotators[mask_1], gamma_vals[mask_1])
        
        mask_0 = (votes == 0)
        beta_num.scatter_add_(0, annotators[mask_0], 1.0 - gamma_vals[mask_0])

        # Safely divide, avoiding division by zero if an annotator never saw a specific ground truth
        epsilon = 1e-6
        self.alpha = alpha_num / (alpha_den + epsilon)
        self.beta = beta_num / (beta_den + epsilon)
        
        # Clip probabilities to prevent log(0) in the next E-step
        self.alpha = torch.clamp(self.alpha, epsilon, 1.0 - epsilon)
        self.beta = torch.clamp(self.beta, epsilon, 1.0 - epsilon)

    def fit(self, prompts: torch.Tensor, annotators: torch.Tensor, votes: torch.Tensor, epochs: int = 20):
        """
        Runs the EM loop until convergence.
        """
        for epoch in range(epochs):
            self.e_step(prompts, annotators, votes)
            self.m_step(prompts, annotators, votes)