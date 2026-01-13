import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from tqdm import tqdm
from spectral_sim import spectral_energy_sim
from Gated_Resnet import GatedResnet
from mahalanobis_filter import MahalanobisFilter
from ProjectUtils import restitch

class ActiveLearningModel:
    """
    Desc: For a set of displacements found in the higher level simulation, active learning will decide, based on the geometric distance between 
        the displacement vector it is looking at and all displacement vectors seen by the model, whether the models prediction can be trusted and therfore output
        or if the model is not confident in its prediction, in which case it will query an FEM solver and train on the result. Over time the model should see many
        displacement vectors and so should not have to query often. This active learning approach uses a vectorised method meaning that the model will not have the 
        benefit of having seen any similar vectors that may reside in the same prediction step, this will lead to a greater number of simulations needing to be run
        but benefits from all steps being done in parrallel.
    Notes: Class functions are vectorised and jit compiled. For this active learning approach to function effectively the model must be 'kick-started' with
        a more general dataset to allow the model to only finetune its predictions during time sensitive deployment of this model. To achieve maximum performance the 
        mesh used must be of equal size throughout all samples it sees during deployment as to maximise the effect of XLA jit compilation.
    Input: 1. Array of all diaplacement vectors in the current prediction step: jax.Array, 2. Mesh of elements being analysed.
    Output: 1. Energy prediction array, 2. Energy sensitivity prediction array.
    Secondary effects: Model will be trained on any data outside a defined confidence bound, data will be generated via a FEM simulation and added to the datastore. 
    """
    def __init__(
            self, 
            confidence_bound, 
            tx: optax.GradientTransformation, 
            learn_rate, 
            epochs, 
            alpha,
            gamma,
            polynomial_order,
            encoder_dim,
            GRN_blocks,
            filter_1_lr,
            filter_1_threshold,
            filter_2_lr,
            filter_2_threshold,
            rngs
        ):
        output_dim = 1
        self.Model = GatedResnet(
            polynomial_order * 3,
            encoder_dim,
            output_dim,
            GRN_blocks,
            mu = jnp.zeros(polynomial_order),
            sigma = jnp.ones(polynomial_order),
            filter_lr=filter_2_lr,
            filter_threshold=filter_2_threshold,
            rngs = rngs
        )
        self.M_input_filter = MahalanobisFilter(
            filter_1_lr,
            filter_1_threshold,
            polynomial_order * 3
        )
        self.optimiser = nnx.Optimizer(self.Model, tx, wrt=nnx.Param)
        self.bound = confidence_bound
        self.LR = learn_rate
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma

    def kick_start(self, Dataset):
        mu_data, sigma_data = Dataset.statistics
        self.Model.mu = mu_data
        self.Model.sigma = sigma_data
        print("Kickstarting Model ...")
        self.Learn(Dataset.batches.coefficients, Dataset.batches.energy, Dataset.batches.e_prime)
        print("Kickstart Complete.")
        
    def query_or_not(self, Coefficients):
        "Returns an array of indicies which correspond to displacement vectors that should and shouldnt be queried"
        query_array = self.M_input_filter.filter(Coefficients)
        should_query = jnp.where(query_array)[0]
        not_query = jnp.where(~query_array)[0]
        return should_query, not_query
    
    def loss_fn(self, target_e_batch, target_e_prime_batch, e_pred_batch, e_prime_pred_batch):
        "Defined loss function: uses MSE for both the energy and energy sensitivity"
        loss_e = jnp.mean((e_pred_batch - target_e_batch)**2)
        loss_e_prime = jnp.mean((e_prime_pred_batch - target_e_prime_batch)**2)
        return self.alpha * loss_e + self.gamma * loss_e_prime
    
    @nnx.jit
    def train_step(self, target_e_batch, target_e_prime_batch, Coefficient_batch):
        "Defines one step in which a batch is processed and the model weights are updated, jit compiled for speed"
        def wrapped_loss(Model):
            e_pred_batch, e_prime_pred_batch, _ = Model(Coefficient_batch)
            loss = self.loss_fn(
                target_e_batch,
                target_e_prime_batch,
                e_pred_batch,
                e_prime_pred_batch
            )
            return loss
    
        grads = nnx.grad(wrapped_loss, argnums=0)(self.Model)
        self.optimiser.update(self.Model, grads)
    
    def Learn(self, coefficient_batch, target_energy_batch, target_derivative_batch):
        "Iterates trainstep for a defined number of steps"
        for _ in tqdm(range(self.epochs), desc="Training Model", leave=False):
            self.train_step(
                target_energy_batch, 
                target_derivative_batch, 
                coefficient_batch
            )
    
    def query_simulator(self, Coefficients): 
        """calls a jax_fem simulation when queried by the model"""
        vmapped_sim_fn = jax.vmap(fun=spectral_energy_sim, in_axes=(None, None, 0))
        e_batch, e_prime_batch = vmapped_sim_fn(
            Coefficients
        )
        return e_batch, e_prime_batch
        
    def __call__(self, Coefficients: jax.Array) -> jax.Array:
        """
        Main call: queries the FEM solver for all samples outside the geometric confidence bound and trains on the data,
            and all samples within the confidence bound are processed by the model. The predictions from the Model and simulation
            are then stitched together and returned.
        """
        query_idx, confident_idx = self.query_or_not(Coefficients)
        E, dE_dC, valid = self.Model(Coefficients[confident_idx])
        query_idx = jnp.concatenate(query_idx, jnp.where(confident_idx, ~valid))
        
        E_sim, dE_dC_sim = self.query_simulator(Coefficients) 
        self.Learn(Coefficients[query_idx], E_sim, dE_dC_sim)

        E = restitch(jnp.where(confident_idx, valid), query_idx, E, E_sim)
        dE_dC = restitch(jnp.where(confident_idx, valid), query_idx, dE_dC, dE_dC_sim)
        return E, dE_dC