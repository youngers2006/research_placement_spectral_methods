import jax
import jax.numpy as jnp

def batch_dataset(dataset_dict, batch_size, train_split, CV_split, test_split, permutated_index_list):
    total_samples = permutated_index_list.shape[0]
    idx_train_samples = int(train_split * total_samples) 
    idx_test_samples = idx_train_samples + int(test_split * total_samples) 

    train_idx = list(permutated_index_list[:idx_train_samples])
    test_idx = list(permutated_index_list[idx_train_samples:idx_test_samples])
    CV_idx = list(permutated_index_list[idx_test_samples:])

    def batch_indices(idx):  
        if not idx:
            return
        num_samples = len(idx)
        num_batches = num_samples // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_idx = idx[start:end]
            
            displacements_batch = [dataset_dict['displacements'][i] for i in batch_idx]
            e_batch = [dataset_dict['target_e'][i] for i in batch_idx]
            e_prime_batch = [dataset_dict['target_e_prime'][i] for i in batch_idx]
            bd_batch = [dataset_dict['boundary_displacements'][i] for i in batch_idx]

            batched_displacements = jnp.array(displacements_batch)
            batched_e = jnp.array(e_batch)
            batched_e_prime = jnp.array(e_prime_batch)
            batched_bd = jnp.array(bd_batch)

            yield {
                'displacements': batched_displacements, 
                'target_e': batched_e, 
                'target_e_prime': batched_e_prime,
                'boundary_displacements': batched_bd
            }
    
    train_batches = list(batch_indices(train_idx))
    test_batches = list(batch_indices(test_idx))
    CV_batches = list(batch_indices(CV_idx))

    return train_batches, CV_batches, test_batches