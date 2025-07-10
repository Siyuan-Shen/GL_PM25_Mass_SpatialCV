import numpy as np

def normalize_Func(inputarray:np.array):
    def _compute_mean_std_batched(array, batch_size=10000):
        n = array.shape[0]
        mean = 0.0
        M2 = 0.0
        count = 0

        for i in range(0, n, batch_size):
            batch = array[i:i+batch_size].astype(np.float32)
            batch_mean = np.mean(batch, axis=0)
            batch_var = np.var(batch, axis=0)
            batch_count = batch.shape[0]

            delta = batch_mean - mean
            total_count = count + batch_count
            mean += delta * batch_count / total_count
            M2 += batch_var * batch_count + (delta ** 2) * count * batch_count / total_count
            count = total_count

        std = np.sqrt(M2 / count) + 1e-5
        return mean, std

    input_mean, input_std = _compute_mean_std_batched(inputarray)
    inputarray -= input_mean
    inputarray /= input_std
    return inputarray,input_mean,input_std



def get_trainingdata_within_sart_end_YEAR(initial_array,training_start_YYYY, training_end_YYYY, start_YYYY, sitesnumber):
    final_array = initial_array[(training_start_YYYY-start_YYYY)*12*sitesnumber:(training_end_YYYY-start_YYYY+1)*12*sitesnumber,:,:,:]
    return final_array