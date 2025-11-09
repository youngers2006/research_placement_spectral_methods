import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from tqdm import tqdm
from spectral_sim import spectral_energy_sim
from fourier_neural_operator import FourierNeuralOperator


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
            X_dim,
            Y_dim, 
            Z_dim, 
            physical_channels, 
            latent_channels, 
            encoder_h1, 
            encoder_h2, 
            num_fno_layers, 
            modes_x, 
            modes_y, 
            modes_z, 
            decoder_h1, 
            decoder_h2, 
            rngs
        ):
        out_channels = 1
        self.Model = FourierNeuralOperator(
            X_dim,
            Y_dim, 
            Z_dim, 
            physical_channels, 
            latent_channels,
            out_channels, 
            encoder_h1, 
            encoder_h2, 
            num_fno_layers, 
            modes_x, 
            modes_y, 
            modes_z, 
            decoder_h1, 
            decoder_h2, 
            rngs
        )
        self.optimiser = nnx.Optimizer(self.Model, tx, wrt=nnx.Param)
        self.bound = confidence_bound
        self.LR = learn_rate
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
    
    def query_or_not(self, energy_variance):
        "Returns an array of indicies which correspond to displacement vectors that should and shouldnt be queried"
        query_array = (energy_variance > self.bound)
        should_query = jnp.where(query_array)[0]
        not_query = jnp.where(~query_array)[0]
        return should_query, not_query
    
    def create_grids(self, boundary_displacements: jax.Array):
        "Creates a list of jraph graphs that can be analysed by the Model"
        grids = 0
        return grids
    
    def loss_fn(self, target_e_batch, target_e_prime_batch, e_pred_batch, e_prime_pred_batch):
        "Defined loss function: uses MSE for both the energy and energy sensitivity"
        loss_e = jnp.mean((e_pred_batch - target_e_batch)**2)
        loss_e_prime = jnp.mean((e_prime_pred_batch - target_e_prime_batch)**2)
        return (self.alpha * loss_e + self.gamma * loss_e_prime)
    
    @nnx.jit
    def train_step(self, target_e_batch, target_e_prime_batch, grid_batch):
        "Defines one step in which a batch is processed and the model weights are updated, jit compiled for speed"
        def wrapped_loss(Model):
            e_pred_batch, e_prime_pred_batch = Model(grid_batch)
            loss = self.loss_fn(
                target_e_batch,
                target_e_prime_batch,
                e_pred_batch,
                e_prime_pred_batch
            )
            return loss
    
        grads = nnx.grad(wrapped_loss, argnums=0)(self.Model)
        self.optimiser.update(self.Model, grads)
    
    def Learn(self, applied_displacement_grids, target_e_from_sim, target_e_prime_from_sim):
        "Iterates trainstep for a defined number of steps"
        for _ in tqdm(range(self.epochs), desc="Training model on simulation data", leave=False):
            self.train_step(
                self.Model, 
                target_e_from_sim, 
                target_e_prime_from_sim, 
                applied_displacement_grids
            )
    
    def query_simulator(self, applied_displacements, grid): 
        """calls a jax_fem simulation when queried by the model"""
        vmapped_sim_fn = jax.vmap(fun=spectral_energy_sim, in_axes=(None, None, 0))
        e_batch, e_prime_batch = vmapped_sim_fn(
            grid, 
            boundary_nodes, 
            applied_displacements
        )
        return e_batch, e_prime_batch
        
    def __call__(self, applied_displacements: jax.Array, grid) -> jax.Array:
        """
        Main call: queries the FEM solver for all samples outside the geometric confidence bound and trains on the data,
            and all samples within the confidence bound are processed by the model. The predictions from the Model and simulation
            are then stitched together and returned.
        """
        applied_displacement_grids = self.create_grids(applied_displacements)
        E, dE_du, E_var = self.Model(applied_displacement_grids)
        query_idx, _ = self.query_or_not(E_var)
        
        E_sim, dE_du_sim = self.query_simulator(applied_displacements[query_idx], grid) 
        self.Learn(self.Model, applied_displacement_graphs_list[query_idx], e_sim, e_prime_sim)

        E = E.at[query_idx].set(E_sim)
        dE_du = dE_du.at[query_idx].set(dE_du_sim)
        return E, dE_du
           

