import os
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx
from flax import struct
from tqdm import tqdm
from typing import Any
from dataclasses import dataclass
from functools import partial
from spectral_sim import spectral_energy_sim


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
    def __init__(self, seen_boundary_displacements, confidence_bound, Model, optimiser, learn_rate, epochs, alpha, gamma):
        self.Model = Model
        self.seen_bds = seen_boundary_displacements
        self.bound = confidence_bound
        self.LR = learn_rate
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.optimiser = optimiser
        
    @jax.jit
    def check_distances(self, applied_displacements):
        "Takes a batch of displacements and compares them to all displacements that have been seen by the model"
        seen_bds = self.seen_bds
        bound = self.bound

        def check_distance(applied_displacement, seen_displacements, bound):
            diff = seen_displacements - applied_displacement
            distances_sq = jnp.sum(jnp.square(diff), axis=(1,2))
            closest_vector_sq = jnp.min(distances_sq)
            should_query = (closest_vector_sq > bound)
            return should_query
        
        vmapped_check = jax.vmap(fun=check_distance, in_axes=(0, None, None))
        should_query_batch = vmapped_check(applied_displacements, seen_bds, bound)
        return should_query_batch
    
    def query_or_not(self, query_array):
        "Returns an array of indicies which correspond to displacement vectors that should and shouldnt be queried"
        should_query = jnp.where(query_array)[0]
        not_query = jnp.where(~query_array)[0]
        return should_query, not_query
    
    def create_graphs(self, boundary_displacements: jax.Array) -> list:
        "Creates a list of jraph graphs that can be analysed by the Model"
        graphs = []
        boundary_nodes = self.Model.boundary_nodes
        base_nodes = self.Model.base_graph.nodes
        base_graph = self.Model.base_graph

        def create_graph(boundary_displacement, nodes, _graph, boundary_idx):
            new_nodes = nodes.at[boundary_idx].set(boundary_displacement)
            graph = _graph.replace(nodes=new_nodes)
            return graph
        
        create_graph_vmapped = jax.vmap(fun=create_graph, in_axes=(0, None, None, None))
        graphs = jraph.unbatch(create_graph_vmapped(boundary_displacements, base_nodes, base_graph, boundary_nodes))
        return graphs
    
    def loss_fn(self, target_e_batch, target_e_prime_batch, e_pred_batch, e_prime_pred_batch):
        "Defined loss function: uses MSE for both the energy and energy sensitivity"
        loss_e = jnp.mean((e_pred_batch - target_e_batch)**2)
        loss_e_prime = jnp.mean((e_prime_pred_batch - target_e_prime_batch)**2)
        return (self.alpha * loss_e + self.gamma * loss_e_prime)
    
    @nnx.jit
    def train_step(self, target_e_batch, target_e_prime_batch, graphs_batch):
        "Defines one step in which a batch is processed and the model weights are updated, jit compiled for speed"
        def wrapped_loss(Model):
            e_pred_batch, e_prime_pred_batch = Model(graphs_batch)
            loss = self.loss_fn(
                target_e_batch,
                target_e_prime_batch,
                e_pred_batch,
                e_prime_pred_batch
            )
            return loss
    
        grads = nnx.grad(wrapped_loss, argnums=0)(self.Model)
        self.optimiser.update(self.Model, grads)
    
    def Learn(self, applied_displacement_graphs_list: jraph.GraphsTuple, target_e_from_sim, target_e_prime_from_sim):
        "Iterates trainstep for a defined number of steps"
        for _ in tqdm(range(self.epochs), desc="Training model on simulation data", leave=False):
            self.train_step(
                self.Model, 
                target_e_from_sim, 
                target_e_prime_from_sim, 
                applied_displacement_graphs_list
            )
    
    def query_simulator(self, applied_displacements, Mesh): 
        """calls a jax_fem simulation when queried by the model"""
        vmapped_sim_fn = jax.vmap(fun=, in_axes=(None, None, 0))
        e_batch, e_prime_batch = vmapped_sim_fn(
            Mesh, 
            self.Model.boundary_nodes, 
            applied_displacements
        )
        return e_batch, e_prime_batch
        
    def __call__(self, applied_displacements: jax.Array, Mesh) -> jax.Array:
        """
        Main call: queries the FEM solver for all samples outside the geometric confidence bound and trains on the data,
            and all samples within the confidence bound are processed by the model. The predictions from the Model and simulation
            are then stitched together and returned.
        """
        should_query = self.check_distance(applied_displacements)
        applied_displacement_graphs_list = self.create_graphs(applied_displacements) 
        query_idx, confident_idx = self.query_or_not(should_query)
        
        e_sim, e_prime_sim = self.query_fem(applied_displacements[query_idx], Mesh) 
        self.Learn(self.Model, applied_displacement_graphs_list[query_idx], e_sim, e_prime_sim)
        
        e_scaled, e_prime_scaled = self.Model.call_single(applied_displacement_graphs_list)
        e_predicted, e_prime_predicted = self.Model.unscale_predictions(e_scaled, e_prime_scaled)

        e_out = restitch(query_idx, confident_idx, e_sim, e_predicted)
        e_prime_out = restitch(query_idx, confident_idx, e_prime_sim, e_prime_predicted)
        return e_out, e_prime_out
           

